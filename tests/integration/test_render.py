"""Forward behaviour of ``splax.render`` and ``splax.render_log``.

The render entry point is walked through its regular invocation, its vmapped and broadcast forms,
and a perceptual comparison against gsplat's ``rasterization``.

Batching must NOT change the blend order within an image. The global sort packs the image id above
the tile bits, so each image's (tile, depth) ordering is identical to the unbatched sort and the
front-to-back accumulation is bit-identical. The vmap tests therefore require **bit-exact** equality
against the loop of unbatched calls, not merely an allclose tolerance. Bit-exactness holds for the
64-bit sort key, which the ``faithful_64bit_keys`` fixture pins. The packed 32-bit key sizes its
depth field from the batch size, so its batched output matches the stack of unbatched renders only
up to a perceptual bound.

gsplat cannot be matched bit-for-bit either, since it uses a different sort, blend, and tiling, so
those differences are bounded perceptually as well. Convention conversions for the gsplat reference
are documented in tests/_gsplat.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import warp as wp
from utils import VIEWMAT, VIEWS, camera, scene

import splax
import splax._cache as _cache

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from types import ModuleType

N = 8_000
KW = {"background": jnp.zeros(3), **camera(128, 128)}

# region single render


def test_render():
    """Render a random scene and check the image against the invariants of the blend."""
    means, scales, quats, colors, opacities, background = scene(20_000, seed=1, dense=True)
    kw = {"viewmat": VIEWMAT, "background": background, **camera(128, 128)}
    image, alpha = splax.render(means, scales, quats, colors, opacities, **kw)
    assert image.shape == (128, 128, 3) and alpha.shape == (128, 128)
    image = np.asarray(image)
    assert np.isfinite(image).all()
    # the blend is convex over the colors and the background, all of which are drawn in [0, 1]
    assert image.min() >= 0.0 and image.max() <= 1.0
    assert not np.allclose(image, np.asarray(background)), "the splat is invisible"
    # pushing every gaussian behind the near plane leaves the pure background
    behind = splax.render(means.at[:, 2].add(-1e3), scales, quats, colors, opacities, **kw)[0]
    np.testing.assert_array_equal(
        np.asarray(behind), np.broadcast_to(np.asarray(background), image.shape)
    )


def test_render_log():
    """Match render_log against the render of the explicitly mapped splat."""
    means, log_scales, quats, logit_colors, logit_opacities, background = scene(4_000, seed=4)
    kw = {"viewmat": VIEWMAT, "background": background, **camera(128, 128)}
    image, _ = splax.render_log(means, log_scales, quats, logit_colors, logit_opacities, **kw)
    reference, _ = splax.render(
        means,
        jnp.exp(log_scales),
        quats,
        jax.nn.sigmoid(logit_colors),
        jax.nn.sigmoid(logit_opacities),
        **kw,
    )
    np.testing.assert_array_equal(np.asarray(image), np.asarray(reference))


def test_render_depth():
    """Render the expected depth map in the fourth image channel.

    The splat sits around the world origin five units in front of the camera, so the covered pixels
    must report camera depths on that order. The depth is a metric camera distance and leaves the
    [0, 1] range a coverage would live in, which is what separates it from the accumulated alpha.
    """
    means, scales, quats, colors, opacities, background = scene(N, seed=5, dense=True)
    kw = {"viewmat": VIEWMAT, "background": background, **camera(128, 128)}
    image, alpha = splax.render(means, scales, quats, colors, opacities, render_depth=True, **kw)
    assert image.shape == (128, 128, 4)
    depth = np.asarray(image)[..., 3]
    covered = np.asarray(alpha) > 0.0
    assert np.isfinite(depth).all()
    assert (depth >= 0.0).all(), "the expected depth is a positive weighted mean of camera depths"
    np.testing.assert_array_equal(depth > 0.0, covered)
    # VIEWMAT places the camera five units from the scene, so the depths land in metric camera units
    # well outside the [0, 1] a coverage is bounded to
    assert depth[covered].max() > 1.0, "the depth is bounded like a coverage"
    assert 4.0 < depth[covered].mean() < 6.0, "the depth is not on the order of the scene distance"
    # the depth accumulator is a separate kernel and must not perturb the color blend
    plain, plain_alpha = splax.render(means, scales, quats, colors, opacities, **kw)
    np.testing.assert_array_equal(np.asarray(image)[..., :3], np.asarray(plain))
    np.testing.assert_array_equal(np.asarray(alpha), np.asarray(plain_alpha))


def test_render_antialiased_changes_output():
    """Check that the Mip-Splatting opacity compensation moves the rendered image."""
    means, scales, quats, colors, opacities, background = scene(2_500, seed=3)
    kw = {"viewmat": VIEWMAT, "background": background, **camera(110, 110)}
    plain = np.asarray(splax.render(means, scales, quats, colors, opacities, **kw)[0])
    aa = np.asarray(
        splax.render(means, scales, quats, colors, opacities, antialiased=True, **kw)[0]
    )
    assert np.abs(aa - plain).max() > 1e-3, "antialiased render must differ from plain"


def test_render_jit_matches_eager():
    """Match splax.render under jit against the eager render byte for byte."""
    splats = scene(50_000, seed=99, dense=True)[:5]
    kw = {"viewmat": VIEWMAT, "background": jnp.zeros(3), **camera(256, 256)}
    eager = np.asarray(splax.render(*splats, **kw)[0])
    jitted = np.asarray(jax.jit(lambda *x: splax.render(*x, **kw)[0])(*splats))
    np.testing.assert_array_equal(eager, jitted)


# region batched render


@pytest.mark.usefixtures("faithful_64bit_keys")
def test_render_vmap_matches_loop():
    """Match vmap over batched splats and viewmats against the unbatched loop, image and alpha."""
    scenes = [scene(N, seed=s, dense=True)[:5] for s in (6, 7, 8)]
    batched = [jnp.stack([sc[i] for sc in scenes]) for i in range(5)]
    ref = [splax.render(*scenes[i], viewmat=VIEWS[i], render_depth=True, **KW) for i in range(3)]
    out = jax.vmap(
        lambda m, s, q, c, o, vm: splax.render(m, s, q, c, o, viewmat=vm, render_depth=True, **KW)
    )(*batched, VIEWS)
    assert out[0].shape == (3, 128, 128, 4) and out[1].shape == (3, 128, 128)
    for k in range(2):  # 0 = the RGB and depth image, 1 = the accumulated alpha
        stacked = jnp.stack([r[k] for r in ref])
        assert out[k].shape == stacked.shape
        np.testing.assert_array_equal(np.asarray(out[k]), np.asarray(stacked))


@pytest.mark.usefixtures("faithful_64bit_keys")
def test_render_vmap_jit_matches_loop():
    """Match jit(vmap(render)) over B=3 viewmats against the unbatched loop."""
    m, s, q, c, o, _bg = scene(N, seed=9, dense=True)
    ref = jnp.stack([splax.render(m, s, q, c, o, viewmat=VIEWS[i], **KW)[0] for i in range(3)])
    fn = jax.jit(jax.vmap(lambda vm: splax.render(m, s, q, c, o, viewmat=vm, **KW)[0]))
    np.testing.assert_array_equal(np.asarray(fn(VIEWS)), np.asarray(ref))


@pytest.mark.usefixtures("faithful_64bit_keys")
def test_render_vmap_batch1_matches_unbatched():
    """Match a B=1 vmap against the plain unbatched render."""
    m, s, q, c, o, _bg = scene(N, seed=10, dense=True)
    vm = VIEWMAT.at[0, 3].set(0.15)
    unbatched = splax.render(m, s, q, c, o, viewmat=vm, **KW)[0]
    b1 = jax.vmap(lambda v: splax.render(m, s, q, c, o, viewmat=v, **KW)[0])(vm[None])
    assert b1.shape == (1, *unbatched.shape)
    np.testing.assert_array_equal(np.asarray(b1[0]), np.asarray(unbatched))


@pytest.mark.usefixtures("faithful_64bit_keys")
def test_render_vmap_larger_batch():
    """Render B=8 at a larger resolution, where the image-id/tile-id key packing is stressed."""
    m, s, q, c, o, _bg = scene(12_000, seed=11, dense=True)
    kw = {"background": jnp.zeros(3), **camera(512, 512)}
    views = jnp.stack([VIEWMAT.at[0, 3].set(0.1 * i) for i in range(8)])
    ref = jnp.stack([splax.render(m, s, q, c, o, viewmat=views[i], **kw)[0] for i in range(8)])
    out = jax.jit(jax.vmap(lambda vm: splax.render(m, s, q, c, o, viewmat=vm, **kw)[0]))(views)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(ref))


@pytest.mark.usefixtures("faithful_64bit_keys")
def test_render_log_vmap_matches_loop():
    """Match vmap(render_log) over B=3 viewmats against the unbatched loop."""
    m, log_s, q, logit_c, logit_o, _bg = scene(4_000, seed=13)
    ref = jnp.stack(
        [
            splax.render_log(m, log_s, q, logit_c, logit_o, viewmat=VIEWS[i], **KW)[0]
            for i in range(3)
        ]
    )
    out = jax.vmap(lambda vm: splax.render_log(m, log_s, q, logit_c, logit_o, viewmat=vm, **KW)[0])(
        VIEWS
    )
    np.testing.assert_array_equal(np.asarray(out), np.asarray(ref))


# region broadcast render


@pytest.mark.usefixtures("faithful_64bit_keys")
def test_render_broadcast():
    """Match vmap over B=3 viewmats with a shared splat against the unbatched loop."""
    m, s, q, c, o, _bg = scene(N, seed=2, dense=True)
    ref = jnp.stack([splax.render(m, s, q, c, o, viewmat=VIEWS[i], **KW)[0] for i in range(3)])
    out = jax.vmap(lambda vm: splax.render(m, s, q, c, o, viewmat=vm, **KW)[0])(VIEWS)
    assert out.shape == ref.shape
    np.testing.assert_array_equal(np.asarray(out), np.asarray(ref))


@pytest.mark.usefixtures("faithful_64bit_keys")
def test_render_broadcast_splats():
    """Match vmap over batched splats with a shared viewmat against the unbatched loop."""
    scenes = [scene(N, seed=s, dense=True)[:5] for s in (3, 4, 5)]
    batched = [jnp.stack([sc[i] for sc in scenes]) for i in range(5)]
    vm = VIEWMAT.at[0, 3].set(0.1)
    ref = jnp.stack([splax.render(*scenes[i], viewmat=vm, **KW)[0] for i in range(3)])
    out = jax.vmap(lambda m, s, q, c, o: splax.render(m, s, q, c, o, viewmat=vm, **KW)[0])(*batched)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(ref))


@pytest.mark.usefixtures("faithful_64bit_keys")
def test_render_vmap_nested_grid():
    """Nested vmap over an A splats x B viewmats grid == the A x B double loop."""
    scenes = [scene(N, seed=s, dense=True)[:5] for s in (3, 4, 5)]
    batched = [jnp.stack([sc[i] for sc in scenes]) for i in range(5)]
    B = VIEWS.shape[0]
    ref = jnp.stack(
        [
            jnp.stack([splax.render(*scenes[a], viewmat=VIEWS[b], **KW)[0] for b in range(B)])
            for a in range(len(scenes))
        ]
    )
    grid = jax.vmap(
        lambda m, s, q, c, o: jax.vmap(lambda vm: splax.render(m, s, q, c, o, viewmat=vm, **KW)[0])(
            VIEWS
        )
    )
    out = grid(*batched)
    assert out.shape == ref.shape
    np.testing.assert_array_equal(np.asarray(out), np.asarray(ref))


@pytest.mark.usefixtures("faithful_64bit_keys")
def test_render_broadcast_nested_transforms():
    """Nested vmap with a shared splat, viewmats on both axes, and transforms on the outer axis."""
    m, s, q, c, o, _bg = scene(N, seed=6, dense=True)
    slices = ((0, N // 2), (N // 2, N))
    n_worlds, n_cams = 2, 3
    vms = jnp.stack(
        [
            jnp.stack([VIEWMAT.at[0, 3].set(0.1 * w + 0.03 * cam) for cam in range(n_cams)])
            for w in range(n_worlds)
        ]
    )
    tfs = jnp.stack(
        [jnp.broadcast_to(jnp.eye(4).at[0, 3].set(0.02 * w), (2, 4, 4)) for w in range(n_worlds)]
    )

    def one(vm: jax.Array, tf: jax.Array) -> jax.Array:
        return splax.render(
            m, s, q, c, o, viewmat=vm, **KW, gaussian_transforms=tf, gaussian_slices=slices
        )[0]

    ref = jnp.stack(
        [jnp.stack([one(vms[w, cam], tfs[w]) for cam in range(n_cams)]) for w in range(n_worlds)]
    )
    out = jax.vmap(lambda vmw, tfw: jax.vmap(lambda vm: one(vm, tfw))(vmw))(vms, tfs)
    assert out.shape == ref.shape
    np.testing.assert_array_equal(np.asarray(out), np.asarray(ref))


# region packed 32-bit sort key

# The packed key is the default. It carves its depth field out of the bits the image id and the tile
# id leave, so a B=8 batched render quantizes depth coarser than its B=1 references and the two
# agree perceptually rather than bit-exactly. This test must therefore run WITHOUT the
# faithful_64bit_keys fixture.


def test_render_vmap_packed_matches_loop():
    """Match the packed batched render against the unbatched loop to a perceptual bound."""
    m, s, q, c, o, _bg = scene(12_000, seed=11, dense=True)
    kw = {"background": jnp.zeros(3), **camera(512, 512)}
    views = jnp.stack([VIEWMAT.at[0, 3].set(0.1 * i) for i in range(8)])
    splax.clear_cache()
    ref = np.asarray(
        jnp.stack([splax.render(m, s, q, c, o, viewmat=views[i], **kw)[0] for i in range(8)])
    )  # B=1 renders each pack with depth_bits=21
    splax.clear_cache()
    out = np.asarray(
        jax.jit(jax.vmap(lambda vm: splax.render(m, s, q, c, o, viewmat=vm, **kw)[0]))(views)
    )  # B=8 render packs with depth_bits=18 (image 3 + tile 10)
    assert _cache._scratch_cache[str(wp.get_device("cuda:0"))]["isect_dtype"] == wp.int32
    d = np.abs(out - ref)
    mse = float(np.mean((out - ref) ** 2))
    psnr = 99.0 if mse == 0 else -10 * np.log10(mse)
    assert d.max() < 0.05, f"packed batched vs stacked max abs diff {d.max():.2e}"
    assert psnr > 65, f"packed batched vs stacked PSNR only {psnr:.1f} dB"
    splax.clear_cache()


# region gsplat parity


@pytest.mark.gsplat
@pytest.mark.parametrize("n,H,W", [(20_000, 256, 256), (100_000, 512, 512)])
def test_render_vs_gsplat(n: int, H: int, W: int, gsplat_shim: ModuleType):
    """splax.render vs gsplat.rasterization on the same scene.

    Different kernels (sort order, blend, opacity-aware tiling), so we bound the image difference
    perceptually. splax's tight tiling drops per-gaussian tails already below 1/255, and gsplat's
    classic rasterizer keeps a slightly different set, so a handful of pixels move by a few 1/255.
    The bulk agree to well under that. Empirically PSNR is comfortably above 30 dB across sizes.
    """
    means, scales, quats, colors, opacities, background = scene(n, seed=n, dense=True)
    kw = {"viewmat": VIEWMAT, "background": background, **camera(H, W)}
    splats = (means, scales, quats, colors, opacities)
    a = np.asarray(splax.render(*splats, **kw)[0])
    b, _b_alpha = gsplat_shim.render(*splats, **kw)
    assert a.shape == b.shape
    mse = float(np.mean((a - b) ** 2))
    psnr = -10 * np.log10(mse) if mse > 0 else float("inf")
    # Measured ~100 dB / max abs ~0.003 across these sizes, bounded with margin.
    assert psnr > 60.0, f"splax vs gsplat render PSNR only {psnr:.1f} dB"
    assert np.abs(a - b).max() < 0.03, f"max abs diff {np.abs(a - b).max():.3f}"


@pytest.mark.gsplat
def test_render_vs_gsplat_lego(
    gsplat_shim: ModuleType, lego_meta: dict, lego_view: Callable[[str], np.ndarray], lego_ply: Path
):
    """splax.render vs gsplat.rasterization on the real lego scene, from a dataset pose.

    A realistic-scene perceptual check to complement the random-scene parity and the GT-PSNR
    regression gate (tests/integration/test_lego_regression.py). Different kernels, so bounded by
    PSNR rather than exactly.
    """
    splats = splax.io.load_ply(lego_ply)
    frame = lego_meta["frames"][0]
    gt = lego_view(frame["file_path"])
    H, W = gt.shape[:2]
    ff = float(0.5 * W / np.tan(0.5 * lego_meta["camera_angle_x"]))
    kw = {
        "viewmat": jnp.asarray(splax.utils.nerf_camera(frame["transform_matrix"])),
        "background": jnp.ones(3),
        "img_shape": (H, W),
        "f": (ff, ff),
        "c": (W // 2, H // 2),
    }
    a = np.asarray(splax.render(*splats, **kw)[0])
    b, _b_alpha = gsplat_shim.render(*splats, **kw)
    mse = float(np.mean((a - b) ** 2))
    psnr = -10 * np.log10(mse) if mse > 0 else float("inf")
    # Measured ~82 dB on this pose, bounded well below with margin for scene detail.
    assert psnr > 45.0, f"splax vs gsplat lego render PSNR only {psnr:.1f} dB"
