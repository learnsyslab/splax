"""Test the forward render across plain calls, batching, and the gsplat reference."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import warp as wp
from utils import VIEWMAT, VIEWS, camera, psnr, scene_params

import splax
import splax._cache as _cache

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from types import ModuleType

N = 8_000
B = len(VIEWS)
KW = {"background": jnp.zeros(3), **camera(128, 128)}


# region single render


def test_render():
    """Render a random scene and check the image against the invariants of the blend."""
    *splats, background = scene_params(20_000, seed=1, dense=True)
    render = jax.jit(
        partial(splax.render, viewmat=VIEWMAT, background=background, **camera(128, 128))
    )

    image, alpha = render(*splats)
    assert image.shape == (128, 128, 3) and alpha.shape == (128, 128)
    assert jnp.isfinite(image).all()
    # the blend is convex over the colors and the background, all of which are drawn in [0, 1]
    assert image.min() >= 0.0 and image.max() <= 1.0
    assert not jnp.allclose(image, background), "the splat is invisible"
    # pushing every gaussian behind the near plane leaves the pure background
    behind = render(splats[0].at[:, 2].add(-1e3), *splats[1:])[0]
    np.testing.assert_array_equal(behind, jnp.broadcast_to(background, image.shape))


def test_render_depth():
    """Render the expected depth map in the fourth image channel."""
    *splats, background = scene_params(N, seed=5, dense=True)
    render = jax.jit(
        partial(splax.render, viewmat=VIEWMAT, background=background, **camera(128, 128)),
        static_argnames="render_depth",
    )

    image, alpha = render(*splats, render_depth=True)
    assert image.shape == (128, 128, 4)
    depth = image[..., 3]
    covered = alpha > 0.0
    assert jnp.isfinite(depth).all()
    assert (depth >= 0.0).all(), "the expected depth is a positive weighted mean of camera depths"
    np.testing.assert_array_equal(depth > 0.0, covered)
    # VIEWMAT places the camera five units from the scene, so the depths land in metric camera units
    # well outside the [0, 1] a coverage is bounded to
    assert depth[covered].max() > 1.0, "the depth is bounded like a coverage"
    assert 4.0 < depth[covered].mean() < 6.0, "the depth is not on the order of the scene distance"
    # the depth accumulator is a separate kernel and must not perturb the color blend
    plain, plain_alpha = render(*splats)
    np.testing.assert_array_equal(image[..., :3], plain)
    np.testing.assert_array_equal(alpha, plain_alpha)


def test_render_antialiased_changes_output():
    """Check that the Mip-Splatting opacity compensation moves the rendered image."""
    *splats, background = scene_params(2_500, seed=3)
    render = jax.jit(
        partial(splax.render, viewmat=VIEWMAT, background=background, **camera(110, 110)),
        static_argnames="antialiased",
    )
    plain, antialiased = render(*splats)[0], render(*splats, antialiased=True)[0]
    assert jnp.abs(antialiased - plain).max() > 1e-3, "antialiased render must differ from plain"


def test_render_jit():
    """Test that render traces and runs with the camera arguments declared static."""
    splats = scene_params(4_000, seed=1, dense=True)[:5]
    render = jax.jit(splax.render, static_argnames=("img_shape", "f", "c", "gaussian_slices"))
    image, alpha = jax.block_until_ready(render(*splats, viewmat=VIEWMAT, **KW))
    assert image.shape == (128, 128, 3) and alpha.shape == (128, 128)


# region batched render

# Batching must not change the blend order within an image. The global sort packs the image id
# above the tile bits, so each image keeps the unbatched (tile, depth) ordering and the vmap
# tests can require exact equality against the loop rather than a tolerance.


@pytest.mark.usefixtures("faithful_64bit_keys")
def test_render_vmap_matches_loop():
    """Match vmap over batched splats and viewmats against the unbatched loop, image and alpha."""
    scenes = [scene_params(N, seed=s, dense=True)[:5] for s in (6, 7, 8)]
    batched = [jnp.stack([sc[i] for sc in scenes]) for i in range(5)]
    render = jax.jit(partial(splax.render, render_depth=True, **KW))

    image, alpha = jax.jit(jax.vmap(render))(*batched, viewmat=VIEWS)
    refs = [render(*scene, viewmat=viewmat) for scene, viewmat in zip(scenes, VIEWS)]
    r_image, r_alpha = (jnp.stack(outs) for outs in zip(*refs))
    assert image.shape == (B, 128, 128, 4) and alpha.shape == (B, 128, 128)
    np.testing.assert_array_equal(image, r_image)
    np.testing.assert_array_equal(alpha, r_alpha)


@pytest.mark.usefixtures("faithful_64bit_keys")
def test_render_vmap_batch1_matches_unbatched():
    """Match a B=1 vmap against the plain unbatched render."""
    splats = scene_params(N, seed=10, dense=True)[:5]
    viewmat = VIEWMAT.at[0, 3].set(0.15)
    render = jax.jit(partial(splax.render, *splats, **KW))

    unbatched = render(viewmat=viewmat)[0]
    batch1 = jax.jit(jax.vmap(render))(viewmat=viewmat[None])[0]
    assert batch1.shape == (1, *unbatched.shape)
    np.testing.assert_array_equal(batch1[0], unbatched)


@pytest.mark.usefixtures("faithful_64bit_keys")
def test_render_vmap_larger_batch():
    """Render B=8 at a larger resolution, where the image-id/tile-id key packing is stressed."""
    splats = scene_params(12_000, seed=11, dense=True)[:5]
    render = jax.jit(partial(splax.render, *splats, background=jnp.zeros(3), **camera(512, 512)))
    views = jnp.stack([VIEWMAT.at[0, 3].set(0.1 * i) for i in range(8)])
    ref = jnp.stack([render(viewmat=viewmat)[0] for viewmat in views])

    out = jax.jit(jax.vmap(render))(viewmat=views)[0]
    np.testing.assert_array_equal(out, ref)


# region broadcast render


@pytest.mark.usefixtures("faithful_64bit_keys")
def test_render_broadcast():
    """Match vmap over B viewmats with a shared splat against the unbatched loop."""
    splats = scene_params(N, seed=2, dense=True)[:5]
    render = jax.jit(partial(splax.render, *splats, **KW))
    ref = jnp.stack([render(viewmat=viewmat)[0] for viewmat in VIEWS])

    out = jax.jit(jax.vmap(render))(viewmat=VIEWS)[0]
    assert out.shape == ref.shape
    np.testing.assert_array_equal(out, ref)


@pytest.mark.usefixtures("faithful_64bit_keys")
def test_render_broadcast_splats():
    """Match vmap over batched splats with a shared viewmat against the unbatched loop."""
    scenes = [scene_params(N, seed=s, dense=True)[:5] for s in (3, 4, 5)]
    batched = [jnp.stack([sc[i] for sc in scenes]) for i in range(5)]
    render = jax.jit(partial(splax.render, viewmat=VIEWMAT.at[0, 3].set(0.1), **KW))
    ref = jnp.stack([render(*scene)[0] for scene in scenes])

    out = jax.jit(jax.vmap(render))(*batched)[0]
    np.testing.assert_array_equal(out, ref)


@pytest.mark.usefixtures("faithful_64bit_keys")
def test_render_vmap_nested_grid():
    """Match a nested vmap over an A splats by B viewmats grid against the double loop."""
    scenes = [scene_params(N, seed=s, dense=True)[:5] for s in (3, 4, 5)]
    batched = [jnp.stack([sc[i] for sc in scenes]) for i in range(5)]
    render = jax.jit(partial(splax.render, **KW))
    ref = jnp.stack(
        [jnp.stack([render(*scene, viewmat=viewmat)[0] for viewmat in VIEWS]) for scene in scenes]
    )

    # The inner vmap holds one scene and sweeps the viewmats, the outer one sweeps the scenes.
    grid = jax.jit(jax.vmap(jax.vmap(render, in_axes=(None,) * 5)))
    views = jnp.broadcast_to(VIEWS, (len(scenes), B, 4, 4))
    out = grid(*batched, viewmat=views)[0]
    assert out.shape == ref.shape
    np.testing.assert_array_equal(out, ref)


@pytest.mark.usefixtures("faithful_64bit_keys")
def test_render_broadcast_nested_transforms():
    """Match a nested vmap with viewmats on both axes and transforms on the outer one."""
    splats = scene_params(N, seed=6, dense=True)[:5]
    n_worlds, n_cams = 2, 3
    render = jax.jit(
        partial(splax.render, *splats, gaussian_slices=((0, N // 2), (N // 2, N)), **KW)
    )
    views = jnp.stack(
        [
            jnp.stack([VIEWMAT.at[0, 3].set(0.1 * w + 0.03 * cam) for cam in range(n_cams)])
            for w in range(n_worlds)
        ]
    )
    transforms = jnp.stack(
        [jnp.broadcast_to(jnp.eye(4).at[0, 3].set(0.02 * w), (2, 4, 4)) for w in range(n_worlds)]
    )
    ref = jnp.stack(
        [
            jnp.stack(
                [
                    render(viewmat=views[w, cam], gaussian_transforms=transforms[w])[0]
                    for cam in range(n_cams)
                ]
            )
            for w in range(n_worlds)
        ]
    )

    # Both mapped operands arrive as keywords, so the transforms are broadcast onto the camera axis.
    grid = jax.jit(jax.vmap(jax.vmap(render)))
    out = grid(
        viewmat=views,
        gaussian_transforms=jnp.broadcast_to(transforms[:, None], (n_worlds, n_cams, 2, 4, 4)),
    )[0]
    assert out.shape == ref.shape
    np.testing.assert_array_equal(out, ref)


# region packed 32-bit sort key

# The packed key is the default. It carves its depth field out of the bits the image id and the tile
# id leave, so a B=8 batched render quantizes depth coarser than its B=1 references and the two
# agree perceptually rather than exactly. This test must therefore run WITHOUT the
# faithful_64bit_keys fixture.


def test_render_vmap_packed_matches_loop():
    """Match the packed batched render against the unbatched loop to a perceptual bound."""
    splats = scene_params(12_000, seed=11, dense=True)[:5]
    render = jax.jit(partial(splax.render, *splats, background=jnp.zeros(3), **camera(512, 512)))
    views = jnp.stack([VIEWMAT.at[0, 3].set(0.1 * i) for i in range(8)])

    splax.clear_cache()
    ref = jnp.stack([render(viewmat=viewmat)[0] for viewmat in views])  # B=1 packs depth_bits=21
    splax.clear_cache()
    out = jax.jit(jax.vmap(render))(viewmat=views)[0]  # B=8 packs image 3 + tile 10, depth_bits=18
    assert _cache._scratch_cache[str(wp.get_device("cuda:0"))]["isect_dtype"] == wp.int32

    deviation, quality = jnp.abs(out - ref).max(), psnr(out, ref)
    assert deviation < 0.05, f"packed batched vs stacked max abs diff {deviation:.2e}"
    assert quality > 65, f"packed batched vs stacked PSNR only {quality:.1f} dB"
    splax.clear_cache()


# region gsplat parity


@pytest.mark.gsplat
@pytest.mark.parametrize("n,H,W", [(20_000, 256, 256), (100_000, 512, 512)])
def test_render_vs_gsplat(n: int, H: int, W: int, gsplat_shim: ModuleType):
    """Bound the render and its alpha against the gsplat rasterization of the same random scene."""
    *splats, background = scene_params(n, seed=n, dense=True)
    means, log_scales, quats, sh_colors, logit_opacities = splats
    scales, colors, opacities = splax.io.apply_activations(log_scales, sh_colors, logit_opacities)
    kw = {"viewmat": VIEWMAT, "background": background, **camera(H, W)}

    img, alpha = jax.jit(partial(splax.render, **kw))(*splats)
    ref_img, ref_alpha = gsplat_shim.render(means, scales, quats, colors, opacities, **kw)
    assert img.shape == ref_img.shape and alpha.shape == ref_alpha.shape
    assert alpha.min() >= 0.0 and alpha.max() <= 1.0
    img_psnr, alpha_psnr = psnr(img, ref_img), psnr(alpha, ref_alpha)
    img_dev = jnp.abs(img - ref_img).max()
    alpha_dev = jnp.abs(alpha - ref_alpha).max()
    # Measured ~100 dB and a max abs difference ~0.003 across these sizes
    assert img_psnr > 60.0, f"splax vs gsplat image PSNR only {img_psnr:.1f} dB"
    assert alpha_psnr > 60.0, f"splax vs gsplat alpha PSNR only {alpha_psnr:.1f} dB"
    assert img_dev < 0.03, f"image max abs diff {img_dev:.3f}"
    assert alpha_dev < 0.03, f"alpha max abs diff {alpha_dev:.3f}"


@pytest.mark.gsplat
@pytest.mark.parametrize("n,H,W", [(20_000, 256, 256), (100_000, 512, 512)])
def test_render_depth_vs_gsplat(n: int, H: int, W: int, gsplat_shim: ModuleType):
    """Bound the image and the expected depth map against the gsplat depth rasterization."""
    *splats, background = scene_params(n, seed=n, dense=True)
    means, log_scales, quats, sh_colors, logit_opacities = splats
    scales, colors, opacities = splax.io.apply_activations(log_scales, sh_colors, logit_opacities)
    kw = {"viewmat": VIEWMAT, "background": background, **camera(H, W)}

    render = jax.jit(partial(splax.render, **kw), static_argnames="render_depth")
    img, _ = render(*splats, render_depth=True)
    ref, _ = gsplat_shim.render_depth(means, scales, quats, colors, opacities, **kw)
    depth, ref_depth = img[..., 3], ref[..., 3]
    assert img.shape == ref.shape
    img_psnr = psnr(img[..., :3], ref[..., :3])
    img_dev = jnp.abs(img[..., :3] - ref[..., :3]).max()
    assert img_psnr > 60.0, f"splax vs gsplat depth blend PSNR only {img_psnr:.1f} dB"
    assert img_dev < 0.03, f"max abs diff {img_dev:.3f}"
    # Both renders leave the same pixels uncovered and read depth 0 there.
    np.testing.assert_array_equal(depth == 0.0, ref_depth == 0.0)
    # The depth carries camera units, so both bounds are taken against its own range
    scale = ref_depth.max()
    rel = jnp.abs(depth - ref_depth).max() / scale
    depth_psnr = psnr(depth / scale, ref_depth / scale)
    assert rel < 0.01, f"relative max depth diff {rel:.2e}"
    assert depth_psnr > 80.0, f"splax vs gsplat depth PSNR only {depth_psnr:.1f} dB"


@pytest.mark.gsplat
def test_render_vs_gsplat_lego(
    gsplat_shim: ModuleType, lego_meta: dict, lego_view: Callable[[str], np.ndarray], lego_ply: Path
):
    """Bound render against the gsplat rasterization of the real lego scene."""
    splats = splax.io.load_ply(lego_ply)
    means, log_scales, quats, sh_colors, logit_opacities = splats
    scales, colors, opacities = splax.io.apply_activations(log_scales, sh_colors, logit_opacities)
    frame = lego_meta["frames"][0]
    H, W = lego_view(frame["file_path"]).shape[:2]
    focal = float(0.5 * W / np.tan(0.5 * lego_meta["camera_angle_x"]))
    kw = {
        "viewmat": jnp.asarray(splax.utils.nerf_camera(frame["transform_matrix"])),
        "background": jnp.ones(3),
        "img_shape": (H, W),
        "f": (focal, focal),
        "c": (W // 2, H // 2),
    }

    img = jax.jit(partial(splax.render, **kw))(*splats)[0]
    ref, _ = gsplat_shim.render(means, scales, quats, colors, opacities, **kw)
    quality = psnr(img, ref)
    # Measured ~82 dB on this pose, bounded well below to leave room for scene detail.
    assert quality > 45.0, f"splax vs gsplat lego render PSNR only {quality:.1f} dB"
