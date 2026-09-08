"""Fit a fixed set of Gaussians to a COLMAP scene with the splax Warp backend.

Generalized trainer for any COLMAP sparse reconstruction consisting of ``sparse/0`` with
``cameras.bin``, ``images.bin``, ``points3D.bin`` and an ``images/`` folder.

Usage:
    pixi run -e tests python scripts/train_colmap.py --data data/drone --out-ply \
        data/scenes/drone.ply
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import time
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import dm_pix
import imageio.v3 as iio
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optax
from colmap import init_from_points, read_camera, read_reconstruction
from scipy.spatial.transform import RigidTransform as TF
from scipy.spatial.transform import Rotation as R

import splax
from splax import render

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable

matplotlib.use("Agg")

logger = logging.getLogger(__name__)
SPLAT_KEYS = ("means", "log_scales", "quats", "colors_logit", "opac_logit")


def _bilinear_sample(D: jax.Array, uv: jax.Array) -> jax.Array:
    """Bilinearly sample the (H, W) depth map at pixel coords ``uv`` (K, 2) = (x, y)."""
    H, W = D.shape
    x = jnp.clip(uv[:, 0] - 0.5, 0.0, W - 1.0)
    y = jnp.clip(uv[:, 1] - 0.5, 0.0, H - 1.0)
    x0 = jnp.floor(x).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)
    x1 = jnp.minimum(x0 + 1, W - 1)
    y1 = jnp.minimum(y0 + 1, H - 1)
    wx = x - x0
    wy = y - y0
    top = D[y0, x0] * (1.0 - wx) + D[y0, x1] * wx
    bot = D[y1, x0] * (1.0 - wx) + D[y1, x1] * wx
    return top * (1.0 - wy) + bot * wy


# region exposure correction


def init_exposure(ntr: int) -> jax.Array:
    """Per-training-image affine color transforms, identity-initialized."""
    eye = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (ntr, 3, 3))
    off = jnp.zeros((ntr, 3, 1), jnp.float32)
    return jnp.concatenate([eye, off], axis=2)


def apply_exposure(img: jax.Array, affine: jax.Array) -> jax.Array:
    """Apply one image's 3x4 affine color transform to an (H, W, 3) render."""
    M, b = affine[:, :3], affine[:, 3]
    return jnp.einsum("ij,hwj->hwi", M, img) + b


# region pose refinement


def init_pose_deltas(ntr: int) -> jax.Array:
    """Per-training-image 6D pose deltas (axis-angle, translation), zero-initialized."""
    return jnp.zeros((ntr, 6), jnp.float32)


def apply_pose_delta(vm: jax.Array, delta: jax.Array) -> jax.Array:
    """Left-compose a small SE3 delta onto a w2c viewmat: R' = Rd R, t' = Rd t + td.

    Rodrigues with the smooth A = sin(t)/t, B = (1-cos(t))/t^2 parameterization so the
    zero-rotation init has well-defined gradients.
    """
    w, t = delta[:3], delta[3:]
    # Not scipy's Rotation.from_rotvec here: it returns NaN gradients at the zero-vector init.
    # The smooth A/B form below keeps jax.grad finite at theta = 0.
    theta2 = jnp.sum(w * w) + 1e-12
    theta = jnp.sqrt(theta2)
    A = jnp.sin(theta) / theta
    B = (1.0 - jnp.cos(theta)) / theta2
    K = jnp.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])
    Rd = jnp.eye(3) + A * K + B * (K @ K)
    out = jnp.eye(4, dtype=vm.dtype)
    out = out.at[:3, :3].set(Rd @ vm[:3, :3])
    out = out.at[:3, 3].set(Rd @ vm[:3, 3] + t)
    return out


def build_loss_fn(
    camera: dict,
    sh_degree: int,
    ssim_lambda: float,
    opacity_reg: float,
    scale_reg: float,
    opacity_entropy: float,
    flat_reg: float,
    depth_loss: bool,
    depth_lambda: float,
    exp_opt: bool,
    pose_opt: bool,
    pose_reg: float,
) -> Callable:
    """Build the photometric, depth and regularization loss over a batch of views."""

    def per_view(
        p: dict[str, jax.Array],
        aux_p: dict[str, jax.Array] | None,
        gt: jax.Array,
        vm: jax.Array,
        bg: jax.Array,
        vi: jax.Array,
        pts_uv: jax.Array,
        pts_depth: jax.Array,
        pts_mask: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Photometric + depth terms for ONE view (vmapped over the batch axis)."""
        if pose_opt:
            assert aux_p is not None
            dlt = jax.lax.dynamic_index_in_dim(aux_p["pose"], vi, axis=0, keepdims=False)
            vm = apply_pose_delta(vm, dlt)
        splats = render_args(p, sh_degree)
        if depth_loss:
            args = {"viewmat": vm, "background": bg, "render_depth": True, **camera}
            colors, _ = render(*splats, **args)
            img = colors[..., :3]
            dpred = _bilinear_sample(colors[..., 3], pts_uv)
            npts = jnp.sum(pts_mask) + 1e-8
            # per-view scale normalization: divide the L1 residual by the mean target
            # depth so the term is dimensionless / scale-invariant.
            scale = jnp.sum(pts_mask * pts_depth) / npts + 1e-8
            dl = jnp.sum(pts_mask * jnp.abs(dpred - pts_depth)) / npts / scale
        else:
            img, _ = render(*splats, viewmat=vm, background=bg, **camera)
            dl = jnp.array(0.0, jnp.float32)
        if exp_opt:
            assert aux_p is not None
            affine = jax.lax.dynamic_index_in_dim(aux_p["exp"], vi, axis=0, keepdims=False)
            img = apply_exposure(img, affine)
        l1 = jnp.mean(jnp.abs(img - gt))
        dssim = jnp.asarray(1.0 - dm_pix.ssim(img, gt))
        return l1, dssim, dl

    def loss_fn(
        p: dict[str, jax.Array],
        aux_p: dict[str, jax.Array] | None,
        gt: jax.Array,
        vm: jax.Array,
        bg: jax.Array,
        vi: jax.Array,
        pts_uv: jax.Array,
        pts_depth: jax.Array,
        pts_mask: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        gt = gt.astype(jnp.float32) / 255.0  # Fusing conversion into the render is faster
        l1s, dssims, dls = jax.vmap(per_view, in_axes=(None, None, 0, 0, 0, 0, 0, 0, 0))(
            p, aux_p, gt, vm, bg, vi, pts_uv, pts_depth, pts_mask
        )
        l1 = jnp.mean(l1s)  # batch-mean photometric (gsplat)
        loss = (1.0 - ssim_lambda) * l1 + ssim_lambda * jnp.mean(dssims)
        loss = loss + opacity_reg * jnp.mean(jax.nn.sigmoid(p["opac_logit"]))
        loss = loss + scale_reg * jnp.mean(jnp.exp(p["log_scales"]))
        if opacity_entropy > 0:
            # SuGaR-style binarization: drive opacities toward 0 or 1 so gaussians
            # act as opaque surface elements rather than semi-transparent fog.
            a = jax.nn.sigmoid(p["opac_logit"])
            ent = -(a * jnp.log(a + 1e-8) + (1.0 - a) * jnp.log(1.0 - a + 1e-8))
            loss = loss + opacity_entropy * jnp.mean(ent)
        if flat_reg > 0:
            # SuGaR-style flatness: shrink only the smallest axis so gaussians
            # become disks that can align with surfaces.
            loss = loss + flat_reg * jnp.mean(jnp.min(jnp.exp(p["log_scales"]), axis=-1))
        if depth_loss:
            loss = loss + depth_lambda * jnp.mean(dls)
        if pose_opt and pose_reg > 0:
            # L2 anchor on the pose deltas: keeps the train poses in the COLMAP gauge so the
            # fixed held-out poses stay consistent with the reconstructed world.
            assert aux_p is not None
            loss = loss + pose_reg * jnp.mean(aux_p["pose"] ** 2)
        return loss, l1

    return loss_fn


def build_step_fn(
    opt: optax.GradientTransformation,
    loss_fn: Callable,
    aux_tx: optax.GradientTransformation | None,
    batch: int,
) -> Callable:
    """Build the jitted optimizer step, taking the auxiliary tables only when they are trained."""
    if aux_tx is None:

        @jax.jit
        def step(
            p: dict[str, jax.Array],
            opt_state: optax.OptState,
            gt: jax.Array,
            vm: jax.Array,
            bg: jax.Array,
            pts_uv: jax.Array,
            pts_depth: jax.Array,
            pts_mask: jax.Array,
        ) -> tuple[dict[str, jax.Array], optax.OptState, jax.Array]:
            vi = jnp.zeros((batch,), jnp.int32)  # unused when aux_tx is None
            (_, l1), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                p, None, gt, vm, bg, vi, pts_uv, pts_depth, pts_mask
            )
            updates, opt_state = opt.update(grads, opt_state, p)
            return optax.apply_updates(p, updates), opt_state, l1
    else:

        @jax.jit
        def step(
            p: dict[str, jax.Array],
            opt_state: optax.OptState,
            aux_p: dict[str, jax.Array],
            aux_state: optax.OptState,
            gt: jax.Array,
            vm: jax.Array,
            bg: jax.Array,
            vi: jax.Array,
            pts_uv: jax.Array,
            pts_depth: jax.Array,
            pts_mask: jax.Array,
        ) -> tuple[
            dict[str, jax.Array], optax.OptState, dict[str, jax.Array], optax.OptState, jax.Array
        ]:
            (_, l1), (grads, aux_grads) = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(
                p, aux_p, gt, vm, bg, vi, pts_uv, pts_depth, pts_mask
            )
            updates, opt_state = opt.update(grads, opt_state, p)
            aux_updates, aux_state = aux_tx.update(aux_grads, aux_state, aux_p)
            return (
                optax.apply_updates(p, updates),
                opt_state,
                optax.apply_updates(aux_p, aux_updates),
                aux_state,
                l1,
            )

    return step


def _view_depth_targets(
    im: dict,
    vm: np.ndarray,
    id2row: dict[int, int],
    pts_xyz_norm: np.ndarray,
    r: float,
    W: int,
    H: int,
    max_pts: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build depth supervision targets for one view."""
    uv = np.zeros((max_pts, 2), np.float32)
    depth = np.zeros((max_pts,), np.float32)
    mask = np.zeros((max_pts,), np.float32)
    rows = np.array([id2row.get(int(p), -1) for p in im["obs_pid"]], np.int64)
    ok = rows >= 0
    if not ok.any():
        return uv, depth, mask
    X = pts_xyz_norm[rows[ok]]  # (K,3) normalized world points
    z = X @ vm[:3, :3].T + vm[:3, 3]  # camera-space coords
    cam_z = z[:, 2]
    px = im["obs_xy"][ok] * r  # downscaled pixel coords (x, y)
    valid = (cam_z > 1e-3) & (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
    px, cam_z = px[valid], cam_z[valid]
    k = px.shape[0]
    if k == 0:
        return uv, depth, mask
    if k > max_pts:
        sel = rng.choice(k, max_pts, replace=False)
        px, cam_z, k = px[sel], cam_z[sel], max_pts
    uv[:k] = px.astype(np.float32)
    depth[:k] = cam_z.astype(np.float32)
    mask[:k] = 1.0
    return uv, depth, mask


def _rectify(
    images: list[dict],
    intr: tuple[float, float, float, float],
    dist: tuple[float, float, float, float, float],
    size: tuple[int, int],
) -> tuple[tuple[float, float, float, float], tuple[int, int], tuple]:
    """Undistort a COLMAP camera onto an ideal pinhole, rewriting its keypoints to match."""
    fx, fy, cx, cy = intr
    W, H = size
    # OpenCV puts the pixel origin at the top-left corner, COLMAP and splax at its center
    K = np.array([[fx, 0.0, cx - 0.5], [0.0, fy, cy - 0.5], [0.0, 0.0, 1.0]], np.float64)
    D = np.asarray(dist, np.float64)
    # alpha=0 gives the largest rectangle free of blank pixels, so no invalid border is trained
    newK, (rx, ry, rw, rh) = cv2.getOptimalNewCameraMatrix(K, D, (W, H), 0)
    assert rw > 0 and rh > 0, "the camera rectifies to an empty image, check the COLMAP fit"
    map_x, map_y = cv2.initUndistortRectifyMap(K, D, None, newK, (W, H), cv2.CV_32FC1)
    newK[:2, 2] -= (rx, ry)
    splits = np.cumsum([len(im["obs_xy"]) for im in images])[:-1]
    obs = np.concatenate([im["obs_xy"] for im in images]).reshape(-1, 1, 2) - 0.5
    obs = cv2.undistortPoints(obs, K, D, P=newK).reshape(-1, 2) + 0.5
    for im, chunk in zip(images, np.split(obs, splits), strict=True):
        im["obs_xy"] = chunk
    pinhole = (
        float(newK[0, 0]),
        float(newK[1, 1]),
        float(newK[0, 2]) + 0.5,
        float(newK[1, 2]) + 0.5,
    )
    return pinhole, (int(rw), int(rh)), (map_x, map_y, np.s_[ry : ry + rh, rx : rx + rw])


def _filter_views(images: list[dict], min_obs: int, pose_filter: float) -> list[dict]:
    """Drop the views whose pose the reconstruction constrains poorly."""
    if min_obs > 0:
        n_all = len(images)
        images = [im for im in images if len(im["obs_pid"]) >= min_obs]
        logger.info(f"min-obs filter: kept {len(images)}/{n_all} views (>= {min_obs} obs)")
    if pose_filter <= 0:
        return images
    n_all = len(images)
    centers = _camera_centers(images)
    med_step = np.median(np.linalg.norm(np.diff(centers, axis=0), axis=1))
    # Drop views that teleport off the trajectory. The 15-frame window is wide enough to absorb
    # excursions of up to about 5 frames.
    half = 7
    keep = np.empty(n_all, bool)
    for i in range(n_all):
        lo, hi = max(0, i - half), min(n_all, i + half + 1)
        keep[i] = np.linalg.norm(centers[i] - np.median(centers[lo:hi], axis=0)) <= (
            pose_filter * med_step
        )
    images = [im for im, k in zip(images, keep, strict=True) if k]
    logger.info(
        f"pose filter: kept {len(images)}/{n_all} views "
        f"(<= {pose_filter:g} x median step {med_step:.4f} off the median path)"
    )
    return images


def _camera_centers(images: list[dict]) -> np.ndarray:
    """Compute the world-space camera centers of a list of COLMAP images, shape ``(N, 3)``."""
    tvecs = np.array([im["tvec"] for im in images])
    rots = R.from_quat(np.array([im["qvec"] for im in images]), scalar_first=True)
    return TF.from_components(tvecs, rots).inv().translation


def _split_views(
    images: list[dict], eval_every: int, adaptive_views: int, frame_step: int
) -> tuple[list[dict], list[dict]]:
    """Split the views into a held-out benchmark and a thinned training set."""
    eval_images = images[::eval_every]
    train_images = [im for i, im in enumerate(images) if i % eval_every != 0]
    n_all = len(train_images)
    if adaptive_views and n_all > adaptive_views:
        # Path length mixes translation with rotation angle in radians, which contribute image
        # motion of the same order at roughly unit camera distance.
        rots = R.from_quat(np.array([im["qvec"] for im in train_images]), scalar_first=True)
        centers = _camera_centers(train_images)
        step = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(step + (rots[1:] * rots[:-1].inv()).magnitude())])
        targets = np.linspace(0.0, arc[-1], adaptive_views)
        sel = np.unique(np.searchsorted(arc, targets).clip(0, n_all - 1))
        train_images = [train_images[i] for i in sel]
        logger.info(f"adaptive sampling: kept {len(train_images)}/{n_all} train views")
    elif frame_step > 1:
        train_images = train_images[::frame_step]
        logger.info(f"frame-step {frame_step}: kept {len(train_images)}/{n_all} train views")
    return eval_images, train_images


def _load_view(
    im: dict, data_dir: Path, size: tuple[int, int], remap: tuple | None, gauge: tuple
) -> tuple[np.ndarray, np.ndarray]:
    """Read one photo, rectify and box-downsample it, and normalize its pose."""
    H, W = size
    arr = iio.imread(data_dir / "images" / im["name"]).astype(np.float32) / 255.0
    if remap is not None:
        map_x, map_y, crop = remap
        arr = cv2.remap(arr, map_x, map_y, cv2.INTER_LINEAR)[crop]
    Hi, Wi = arr.shape[:2]
    fh, fw = Hi // H, Wi // W
    arr = arr[: H * fh, : W * fw].reshape(H, fh, W, fw, 3).mean((1, 3))
    s, center = gauge
    rmat = R.from_quat(im["qvec"], scalar_first=True).as_matrix()
    vm = np.eye(4, dtype=np.float32)
    vm[:3, :3] = rmat
    vm[:3, 3] = s * (im["tvec"] + rmat @ center)
    return arr, vm


def load_scene(
    data_dir: str | Path,
    downscale: int,
    eval_every: int,
    max_depth_pts: int = 2048,
    seed: int = 0,
    sparse_model: int = 0,
    load_workers: int = 16,
    min_obs: int = 0,
    sparse_dir: str = "sparse",
    pose_filter: float = 0.0,
    frame_step: int = 1,
    adaptive_views: int = 0,
    undistort: bool = True,
) -> dict:
    """Load a COLMAP scene, normalized, downscaled, and by default rectified."""
    data_dir = Path(data_dir)
    # COLMAP can emit several disconnected sub-models, and the largest is not always 0
    cams, images, points = read_reconstruction(data_dir / sparse_dir / str(sparse_model))
    images = _filter_views(images, min_obs, pose_filter)
    pts_xyz, pts_rgb, pts_ids, pts_track_lens = points
    id2row = {int(pid): i for i, pid in enumerate(pts_ids)}
    centers = _camera_centers(images)
    ctr = np.median(centers, axis=0)
    s = 1.0 / np.mean(np.linalg.norm(centers - ctr, axis=1))
    gauge = (s, ctr)
    pts_xyz = (s * (pts_xyz - ctr)).astype(np.float32)
    eval_images, train_images = _split_views(images, eval_every, adaptive_views, frame_step)

    cam_name, W0, H0, params = cams[images[0]["camera_id"]]
    (fx, fy, cx, cy), dist = read_camera(cam_name, params)
    # Rectify at load, where nerfstudio's datamanager also does it, and render an ideal pinhole.
    remap = None
    if undistort and any(dist):
        params, (W0, H0), remap = _rectify(images, (fx, fy, cx, cy), dist, (W0, H0))
        fx, fy, cx, cy = params
        logger.info(f"rectified {cam_name} to PINHOLE {W0}x{H0}")
        cam_name, dist = "PINHOLE", (0.0, 0.0, 0.0, 0.0, 0.0)
    W, H = W0 // downscale, H0 // downscale
    r = W / W0
    # Distortion coefficients live on normalized coordinates and survive the downscale untouched.
    intr = (fx * r, fy * r, cx * r, cy * r)

    n_train, n_eval = len(train_images), len(eval_images)
    # Train images are stored uint8, so the whole set fits in host RAM. The ground truth came from
    # uint8 JPEGs, and the only loss is sub-LSB rounding of the box-downsample mean.
    train_imgs = np.empty((n_train, H, W, 3), np.uint8)
    train_vms = np.empty((n_train, 4, 4), np.float32)
    eval_imgs = np.empty((n_eval, H, W, 3), np.float32)
    eval_vms = np.empty((n_eval, 4, 4), np.float32)
    tp_uv = np.empty((n_train, max_depth_pts, 2), np.float32)
    tp_depth = np.empty((n_train, max_depth_pts), np.float32)
    tp_mask = np.empty((n_train, max_depth_pts), np.float32)
    tgt_rng = np.random.default_rng(seed)
    load = partial(_load_view, data_dir=data_dir, size=(H, W), remap=remap, gauge=gauge)
    n_workers = min(max(1, load_workers), n_train + n_eval)
    logger.info(
        f"loading {n_train} train / {n_eval} eval images at {W}x{H} "
        f"(downscale {downscale}, {n_workers} workers) ..."
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        for i, (arr, vm) in enumerate(pool.map(load, eval_images)):
            eval_imgs[i], eval_vms[i] = arr, vm
        for i, (im, (arr, vm)) in enumerate(
            zip(train_images, pool.map(load, train_images), strict=True)
        ):
            train_imgs[i] = np.clip(arr * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
            train_vms[i] = vm
            tp_uv[i], tp_depth[i], tp_mask[i] = _view_depth_targets(
                im, vm, id2row, pts_xyz, r, W, H, max_depth_pts, tgt_rng
            )
    return {
        "train_imgs": train_imgs,
        "train_vms": train_vms,
        "eval_imgs": eval_imgs,
        "eval_vms": eval_vms,
        "eval_names": [im["name"] for im in eval_images],
        "H": H,
        "W": W,
        "intr": intr,
        "dist": dist,
        "pts_xyz": pts_xyz,
        "pts_rgb": pts_rgb,
        "pts_track_lens": pts_track_lens,
        "cam_name": cam_name,
        "cam_params": params,
        "norm_scale": float(s),
        "norm_center": ctr,
        "train_pts_uv": tp_uv,
        "train_pts_depth": tp_depth,
        "train_pts_mask": tp_mask,
    }


# region Rendering / metrics


def render_args(params: dict[str, jax.Array], sh_degree: int) -> tuple[jax.Array, ...]:
    """Map the trainer parameters onto the arguments ``render`` takes.

    The base colour is optimized as a logit, so it stays inside the displayable range.
    """
    base = splax.io.rgb_to_sh(jax.nn.sigmoid(params["colors_logit"]))
    sh_colors = jnp.concatenate([base, params["sh_rest"][:, : (sh_degree + 1) ** 2 - 1]], axis=1)
    return (params["means"], params["log_scales"], params["quats"], sh_colors, params["opac_logit"])


def psnr(a: np.ndarray | jax.Array, b: np.ndarray | jax.Array) -> float:
    """Compute PSNR from two images in [0, 1]."""
    mse = float(np.mean((np.clip(np.asarray(a), 0, 1) - np.asarray(b)) ** 2))
    return -10 * np.log10(mse) if mse > 0 else float("inf")


def save_ply(path: str | Path, params: dict[str, jax.Array], sh_degree: int):
    """Write current parameters to a 3DGS PLY file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    splax.io.write_ply(path, *render_args(params, sh_degree))
    logger.info(f"wrote {path}")


def _reset_opt_state(opt_state: optax.OptState, reset_mask: jax.Array) -> optax.OptState:
    n = reset_mask.shape[0]
    keep = (~reset_mask).astype(jnp.float32)

    def z(x: jax.Array) -> jax.Array:
        if isinstance(x, jnp.ndarray) and x.ndim >= 1 and x.shape[0] == n:
            return x * keep.reshape((-1,) + (1,) * (x.ndim - 1))
        return x

    return jax.tree.map(z, opt_state)


# region Training
def build_optimizer(
    args: argparse.Namespace, params: dict[str, jax.Array], lr_scale: float
) -> tuple[optax.GradientTransformation, optax.Schedule]:
    """Give every parameter group its own Adam rate, and hand back the means schedule."""
    decay_steps = args.decay_steps if args.decay_steps else args.steps
    means_sched = optax.exponential_decay(args.means_lr * lr_scale, decay_steps, 0.01)

    def group_sched(lr: float) -> float | optax.Schedule:
        """Hold a rate flat, or plateau and decay it to a hundredth from --late-decay-start."""
        if not args.late_decay_start:
            return lr
        tail = optax.exponential_decay(lr, max(1, args.steps - args.late_decay_start), 0.01)
        return optax.join_schedules([optax.constant_schedule(lr), tail], [args.late_decay_start])

    txs: dict[Hashable, optax.GradientTransformation] = {
        "means": optax.adam(means_sched),
        "log_scales": optax.adam(group_sched(args.scales_lr * lr_scale)),
        "quats": optax.adam(group_sched(args.quats_lr * lr_scale)),
        "colors_logit": optax.adam(group_sched(args.colors_lr * lr_scale)),
        "opac_logit": optax.adam(group_sched(args.opac_lr * lr_scale)),
        "sh_rest": optax.adam(group_sched(args.colors_lr / 20 * lr_scale)),  # From Inria's recipe
    }
    return optax.multi_transform(txs, {k: k for k in params}), means_sched


def build_aux_optimizer(
    args: argparse.Namespace, ntr: int, lr_scale: float
) -> tuple[optax.GradientTransformation | None, dict[str, jax.Array] | None, optax.OptState | None]:
    """Build the per-image exposure and pose tables, one optax group each, or nothing."""
    txs: dict[Hashable, optax.GradientTransformation] = {}
    aux_params: dict[str, jax.Array] = {}
    if args.exposure_opt:
        txs["exp"] = optax.adam(args.exposure_lr * lr_scale)
        aux_params["exp"] = init_exposure(ntr)
        logger.info("Exposure correction enabled. Learning per-image affine transforms")
    if args.pose_opt:
        txs["pose"] = optax.adam(args.pose_lr * lr_scale)
        aux_params["pose"] = init_pose_deltas(ntr)
        logger.info("Pose refinement enabled. Learning per-image SE3 deltas")
    if not txs:
        return None, None, None
    aux_tx = optax.multi_transform(txs, {k: k for k in aux_params})
    return aux_tx, aux_params, aux_tx.init(aux_params)


def build_relocate_fn(args: argparse.Namespace) -> Callable:
    """Build the jitted relocation of the gaussians that went transparent."""
    binoms = splax.mcmc.make_binoms(51)

    @jax.jit
    def relocate(
        p: dict[str, jax.Array], opt_state: optax.OptState, key: jax.Array
    ) -> tuple[dict[str, jax.Array], optax.OptState]:
        colors = jnp.concatenate([p["colors_logit"], p["sh_rest"]], axis=1)
        splats = (p["means"], p["log_scales"], p["quats"], colors, p["opac_logit"])
        new, reset = splax.mcmc.relocate(key, *splats, binoms, min_opacity=args.min_opacity)
        rest = {"colors_logit": new[3][:, :1], "sh_rest": new[3][:, 1:]}
        return dict(zip(SPLAT_KEYS, new)) | rest, _reset_opt_state(opt_state, reset)

    return relocate


def build_inject_noise_fn(args: argparse.Namespace) -> Callable:
    """Build the jitted MCMC noise injection into the gaussian positions."""

    @jax.jit
    def inject_noise(
        p: dict[str, jax.Array], key: jax.Array, scaler: float
    ) -> dict[str, jax.Array]:
        splats = (p["means"], p["log_scales"], p["quats"], p["opac_logit"])
        means = splax.mcmc.inject_noise(key, *splats, scaler, min_opacity=args.min_opacity)
        return {**p, "means": means}

    return inject_noise


def train(args: argparse.Namespace) -> dict:
    """Train splats on a COLMAP scene and return metrics."""
    scene = load_scene(
        args.data,
        args.downscale,
        args.eval_every,
        max_depth_pts=args.max_depth_pts,
        seed=args.seed,
        sparse_model=args.sparse_model,
        load_workers=args.load_workers,
        min_obs=args.min_obs,
        sparse_dir=args.sparse_dir,
        pose_filter=args.pose_filter,
        frame_step=args.frame_step,
        adaptive_views=args.adaptive_views,
        undistort=args.undistort,
    )
    H, W, intr, dist = scene["H"], scene["W"], scene["intr"], scene["dist"]
    ntr = scene["train_imgs"].shape[0]
    logger.info(f"{ntr} train / {len(scene['eval_names'])} eval views")
    logger.info(f"{scene['pts_xyz'].shape[0]} sparse points -> {args.n} gaussians")

    params = init_from_points(
        scene["pts_xyz"],
        scene["pts_rgb"],
        args.n,
        args.init_opa,
        args.seed,
        weights=scene["pts_track_lens"],
    )
    params["sh_rest"] = jnp.zeros((args.n, (args.sh_degree + 1) ** 2 - 1, 3), jnp.float32)

    # host-side image stacks; move one view per step (keeps GPU memory modest)
    train_imgs = scene["train_imgs"]
    train_vms = jnp.asarray(scene["train_vms"])
    # depth targets stay on the host too, one view moved per step
    tp_uv = scene["train_pts_uv"]
    tp_depth = scene["train_pts_depth"]
    tp_mask = scene["train_pts_mask"]
    if args.depth_loss:
        vis = float(tp_mask.sum(1).mean())
        logger.info(f"Depth regularizer: {vis:.0f}/{args.max_depth_pts} points per train view")

    eval_imgs = [scene["eval_imgs"][i] for i in range(len(scene["eval_names"]))]
    eval_vms = [jnp.asarray(scene["eval_vms"][i]) for i in range(len(eval_imgs))]

    camera: dict = {"img_shape": (H, W), "f": intr[:2], "c": intr[2:], "dist": dist}
    camera |= {"antialiased": args.antialiased}

    def eval_psnr(idxs: list[int]) -> list[float]:
        splats = render_args(params, args.sh_degree)
        white = jnp.ones(3)
        return [
            psnr(render(*splats, viewmat=eval_vms[i], background=white, **camera)[0], eval_imgs[i])
            for i in idxs
        ]

    # spread the scored eval views over the whole trajectory. The first n_eval held-out
    # views all come from the start of the capture and are not representative.
    n_scored = min(args.n_eval, len(eval_imgs))
    eval_idxs = sorted(set(np.linspace(0, len(eval_imgs) - 1, n_scored).astype(int).tolist()))

    # Scale batched learning rates by sqrt(B) and adjust relocation steps
    B = args.batch_size
    lr_scale = float(np.sqrt(B))
    relocate_every = max(1, round(args.relocate_every / B)) if args.relocate_every else 0
    refine_start = round(args.refine_start / B)
    refine_stop = args.refine_stop  # already 0.9*steps in reduced-step units
    noise_stop_iter = (
        round(args.noise_stop_iter / B) if args.noise_stop_iter > 0 else args.noise_stop_iter
    )
    if B > 1:
        logger.info(f"Batched training: LRs scaled to {lr_scale:.3f}, relocate and refine adjusted")

    opt, means_sched = build_optimizer(args, params, lr_scale)
    opt_state = opt.init(params)
    relocate = build_relocate_fn(args)
    inject_noise = build_inject_noise_fn(args)
    aux_tx, aux_params, aux_state = build_aux_optimizer(args, ntr, lr_scale)
    build_loss = partial(
        build_loss_fn,
        camera,
        ssim_lambda=args.ssim_lambda,
        opacity_reg=args.opacity_reg,
        scale_reg=args.scale_reg,
        opacity_entropy=args.opacity_entropy,
        flat_reg=args.flat_reg,
        depth_loss=args.depth_loss,
        depth_lambda=args.depth_lambda,
        exp_opt=args.exposure_opt,
        pose_opt=args.pose_opt,
        pose_reg=args.pose_reg,
    )

    def make(degree: int) -> Callable:
        """Rebuild the step around the harmonics degree the warm-up has reached."""
        return build_step_fn(opt, build_loss(degree), aux_tx, B)

    sh_degree = 0  # Ramp up to args.sh_degree in steps of 1 every args.sh_interval steps
    step_fn = make(sh_degree)

    p0 = float(np.mean(eval_psnr(eval_idxs)))
    logger.info(f"point-init eval PSNR: {p0:.2f} dB")
    curve = [{"step": 0, "visits": 0, "eval_psnr": round(p0, 3)}]

    key = jax.random.key(args.seed + 1)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(ntr)
    white = jnp.ones(3)
    t0 = time.perf_counter()
    for it in range(1, args.steps + 1):
        if sh_degree < min(it // args.sh_interval, args.sh_degree):
            sh_degree += 1
            step_fn = make(sh_degree)
        # B consecutive view visits per step
        vis = [int(order[((it - 1) * B + 1 + j) % ntr]) for j in range(B)]
        vidx = np.asarray(vis)
        gt = jnp.asarray(train_imgs[vidx])  # (B, H, W, 3) uint8, the loss converts it
        vm = train_vms[jnp.asarray(vidx)]  # (B, 4, 4)
        if args.random_bkgd:
            keys = jax.random.split(key, B + 1)  # one background draw per view
            key = keys[0]
            bg = jax.vmap(lambda k: jax.random.uniform(k, (3,)))(keys[1:])
        else:
            bg = jnp.broadcast_to(white, (B, 3))
        pt_args = (
            jnp.asarray(tp_uv[vidx]),
            jnp.asarray(tp_depth[vidx]),
            jnp.asarray(tp_mask[vidx]),
        )
        if aux_tx is not None:
            params, opt_state, aux_params, aux_state, l1 = step_fn(
                params,
                opt_state,
                aux_params,
                aux_state,
                gt,
                vm,
                bg,
                jnp.asarray(vidx, jnp.int32),
                *pt_args,
            )
        else:
            params, opt_state, l1 = step_fn(params, opt_state, gt, vm, bg, *pt_args)

        if relocate_every and refine_start < it < refine_stop and it % relocate_every == 0:
            key, sk = jax.random.split(key)
            params, opt_state = relocate(params, opt_state, sk)
        if args.noise_lr > 0 and it < args.steps and (noise_stop_iter < 0 or it < noise_stop_iter):
            scaler = float(jnp.asarray(means_sched(it))) * args.noise_lr
            key, sk = jax.random.split(key)
            params = inject_noise(params, sk, scaler)

        if it % args.log_every == 0 or it == args.steps:
            l1.block_until_ready()
            ep = float(np.mean(eval_psnr(eval_idxs)))
            curve.append(
                {
                    "step": it,
                    "visits": it * B,
                    "eval_psnr": round(ep, 3),
                    "train_l1": round(float(l1), 5),
                }
            )
            logger.info(f"step {it:5d}  train L1 {float(l1):.4f}  eval PSNR {ep:5.2f} dB")
    wall = time.perf_counter() - t0

    per_frame = eval_psnr(eval_idxs)
    ep_final = float(np.mean(per_frame))
    logger.info(f"\nfinal held-out PSNR: {ep_final:.2f} dB  {[round(x, 2) for x in per_frame]}")
    logger.info(f"{args.steps} steps / {args.n} gaussians in {wall:.1f}s ")

    if args.out_ply:
        save_ply(args.out_ply, params, args.sh_degree)
    if args.plot:
        _plot_curve(curve, wall, ep_final)
    result = {
        "per_frame": per_frame,
        "names": [scene["eval_names"][i] for i in eval_idxs],
        "final": ep_final,
        "wall": wall,
        "curve": curve,
        "batch": B,
        "steps": args.steps,
        "n": args.n,
        "view_visits": args.steps * B,
        "views_per_s": round(args.steps * B / wall, 1),
        "ms_per_step": round(wall / args.steps * 1000, 3),
        "depth_loss": bool(args.depth_loss),
        "depth_lambda": args.depth_lambda,
    }
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"wrote {args.out_json}")
    return result


def _plot_curve(curve: list[dict], wall: float, final: float):
    """Plot and save the held out PSNR curve."""
    steps = [c["step"] for c in curve]
    ps = [c["eval_psnr"] for c in curve]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, ps, "-o", ms=3, color="C0")
    ax.set_xlabel("training step")
    ax.set_ylabel("held-out PSNR (dB)")
    ax.set_title(f"MCMC fit: {final:.2f} dB in {wall:.0f}s")
    ax.grid(alpha=0.3)
    dir = Path("reports/figures")
    dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(dir / "training.png"), dpi=130)
    logger.info(f"wrote {dir / 'training.png'}")


def main():
    """Parse CLI args and run COLMAP training."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="COLMAP scene dir (has sparse/<i>, images/)")
    ap.add_argument(
        "--sparse-model", type=int, default=0, help="COLMAP sub-model index under sparse/ "
    )
    ap.add_argument(
        "--load-workers",
        type=int,
        default=16,
        help="parallel image decode workers (1 minimizes load-time memory growth)",
    )
    ap.add_argument("--out-ply", default="data/scenes/train.ply")
    ap.add_argument("--downscale", type=int, default=4, help="image downscale factor")
    ap.add_argument(
        "--undistort",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="rectify the photos at load and render an ideal pinhole, as nerfstudio does",
    )
    ap.add_argument("--eval-every", type=int, default=8, help="hold out every Nth image")
    ap.add_argument("--n-eval", type=int, default=3, help="held-out views scored/rendered")
    ap.add_argument("--n", type=int, default=150_000)
    ap.add_argument(
        "--sh-degree",
        type=int,
        default=3,
        choices=(0, 1, 2, 3),
        help="spherical harmonics degree for view-dependent color (0=fixed color)",
    )
    ap.add_argument(
        "--sh-interval",
        type=int,
        default=500,
        help="steps between harmonics degree activations during the warm-up",
    )
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="views per training step. The loss is averaged over the batch, every learning rate "
        "is scaled by sqrt(batch) and the MCMC cadence by 1/batch, so --steps is "
        "total_view_visits/batch",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--antialiased",
        action="store_true",
        help="Mip-Splatting opacity compensation (gsplat rasterize_mode=antialiased)",
    )
    ap.add_argument("--log-every", type=int, default=200)
    # 6d MCMC recipe defaults (transferred via scene normalization)
    ap.add_argument(
        "--decay-steps",
        type=int,
        default=None,
        help="horizon for the means-LR (and thus MCMC noise) exponential decay; "
        "defaults to --steps. Set to keep the decay pace when training longer.",
    )
    ap.add_argument(
        "--late-decay-start",
        type=int,
        default=0,
        help="from this step, decay the scales/quats/colors/opacity LRs exponentially to 1%% "
        "by --steps (0 = constant LRs, the base recipe)",
    )
    ap.add_argument("--means-lr", type=float, default=1.5e-3)
    ap.add_argument("--scales-lr", type=float, default=5e-3)
    ap.add_argument("--quats-lr", type=float, default=1e-3)
    ap.add_argument("--colors-lr", type=float, default=1e-2)
    ap.add_argument("--opac-lr", type=float, default=5e-2)
    ap.add_argument("--ssim-lambda", type=float, default=0.2)
    ap.add_argument("--opacity-reg", type=float, default=0.01)
    ap.add_argument("--scale-reg", type=float, default=0.01)
    ap.add_argument(
        "--opacity-entropy",
        type=float,
        default=0.0,
        help="weight of the SuGaR-style opacity binarization term (0=off)",
    )
    ap.add_argument(
        "--flat-reg",
        type=float,
        default=0.0,
        help="weight of the SuGaR-style min-axis scale penalty (0=off)",
    )
    ap.add_argument("--noise-lr", type=float, default=5e5)
    ap.add_argument(
        "--noise-stop-iter",
        type=int,
        default=-1,
        help="stop MCMC noise injection after this step (-1=never, gsplat default)",
    )
    ap.add_argument("--min-opacity", type=float, default=0.005)
    ap.add_argument("--relocate-every", type=int, default=100)
    ap.add_argument("--refine-start", type=int, default=200)
    ap.add_argument("--refine-stop", type=int, default=None, help="default 0.9*steps")
    ap.add_argument("--init-opa", type=float, default=0.1)
    ap.add_argument(
        "--random-bkgd",
        action="store_true",
        help="random per-step render-side bg color (gsplat random_bkgd). "
        "CAVEAT: COLMAP photos carry no alpha, so only the render is "
        "recomposited -- the fixed real GT photo is not. Off by default; "
        "see reports/phase8d_random_bkgd.md.",
    )
    ap.add_argument(
        "--depth-loss",
        action="store_true",
        help="scale-normalized masked L1 between the rendered expected-depth channel and the "
        "COLMAP sparse points' camera depths",
    )
    ap.add_argument(
        "--depth-lambda", type=float, default=1e-2, help="depth-loss weight (gsplat default 1e-2)"
    )
    ap.add_argument(
        "--max-depth-pts",
        type=int,
        default=2048,
        help="fixed max COLMAP sparse points per view for depth reg",
    )
    ap.add_argument(
        "--sparse-dir",
        default="sparse",
        help="name of the sparse reconstruction dir under the scene dir",
    )
    ap.add_argument(
        "--pose-filter",
        type=float,
        default=0.0,
        help="drop views whose camera center deviates from the windowed-median "
        "trajectory by more than this multiple of the median step (0 = off)",
    )
    ap.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="keep every Nth view (video captures are highly redundant)",
    )
    ap.add_argument(
        "--adaptive-views",
        type=int,
        default=0,
        help="keep ~N views sampled uniformly along the camera path (translation + rotation) "
        "instead of uniformly in time; overrides --frame-step (0 = off)",
    )
    ap.add_argument(
        "--min-obs",
        type=int,
        default=0,
        help="drop views with fewer triangulated COLMAP observations (weakly "
        "constrained poses, frequent misregistrations on video captures); 0 keeps all",
    )
    ap.add_argument("--out-json", default=None, help="dump the result dict as JSON")
    ap.add_argument(
        "--exposure-opt",
        action="store_true",
        help="learn a per-training-image affine color correction, so the shared 3D color does not "
        "absorb capture exposure drift as view dependence",
    )
    ap.add_argument(
        "--exposure-lr", type=float, default=1e-3, help="LR for the exposure affine params"
    )
    ap.add_argument(
        "--pose-opt",
        action="store_true",
        help="jointly refine a per-training-view SE3 pose delta. Held-out poses stay fixed, and "
        "the depth targets are built from the unrefined poses, so they go stale as deltas grow",
    )
    ap.add_argument("--pose-lr", type=float, default=1e-4, help="LR for the per-view pose deltas")
    ap.add_argument(
        "--pose-reg",
        type=float,
        default=0.0,
        help="L2 anchor on the pose deltas (gauge stays tied to COLMAP; try 1e-1)",
    )
    ap.add_argument("--no-plot", dest="plot", action="store_false")
    args = ap.parse_args()
    if args.refine_stop is None:
        args.refine_stop = int(0.9 * args.steps)
    train(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("jax").setLevel(logging.WARNING)
    main()
