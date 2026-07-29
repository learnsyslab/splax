"""Intersection sort and binning.

Emits sorted intersection keys and per-tile bin edges. Sorting uses radix sort, which is stable and
deterministic. This allows us to recompute the sorting in the backward pass with identical results
to the forward pass.
"""

import warp as wp

from splax._cache import begin_count_read, cached_launch, cached_scratch, fetch_count_read
from splax._intersect import BLOCK_WIDTH
from splax._rasterize._sort._kernels import (
    MINMAX_CHUNK,
    depth_minmax,
    map_intersects_32bit,
    map_intersects_64bit,
    seed_minmax,
    tile_bin_edges_32bit,
    tile_bin_edges_64bit,
)


def sort_and_bin(
    xys: wp.array,
    depths: wp.array,
    radii: wp.array,
    conics: wp.array,
    map_opacities: wp.array,
    cum_tiles_hit: wp.array,
    n: int,
    B: int,
    img_h: int,
    img_w: int,
) -> tuple[wp.array, wp.array, int, int, int]:
    """Emit sorted intersection keys and tile bins for one batched launch.

    Args:
        xys: (B*n, 2) array of gaussian centers in pixel coordinates.
        depths: (B*n,) array of gaussian depths.
        radii: (B*n,) array of gaussian radii in pixels.
        conics: (B*n, 3) array of gaussian conics.
        map_opacities: (B*n,) array of gaussian opacities.
        cum_tiles_hit: (B*n,) array of cumulative tile counts per gaussian.
        n: The number of gaussians per image.
        B: The number of distinct images in the batch.
        img_h: The image height in pixels.
        img_w: The image width in pixels.

    Returns:
        gaussian_ids: (n_intersects,) array of sorted gaussian indices.
        tile_bins: (B, n_tiles, 2) array of per-tile bin edges.
        n_intersects: The number of intersecting gaussians.
        tile_bounds_x: The number of tile columns.
        n_tiles: The total number of tiles.
    """
    device = xys.device
    tile_bounds_x = (img_w + BLOCK_WIDTH - 1) // BLOCK_WIDTH
    tile_bounds_y = (img_h + BLOCK_WIDTH - 1) // BLOCK_WIDTH
    n_tiles = tile_bounds_x * tile_bounds_y
    # bits to index [0, n_tiles) and [0, B), the tile and image id fields of the sort key
    tile_n_bits = (n_tiles - 1).bit_length()
    image_n_bits = (B - 1).bit_length()
    upper_bits = image_n_bits + tile_n_bits
    depth_bits = 31 - upper_bits
    _32bit_packed = _use_32bit_keys(depth_bits)
    if not _32bit_packed and upper_bits > 32:
        raise ValueError(
            f"batched intersection key overflow: {image_n_bits=}, {tile_n_bits=}, {upper_bits=}. "
            f"Reduce batch size or resolution."
        )
    total = B * n
    bins_len = B * n_tiles
    opac_mod = map_opacities.shape[0]

    # Because the size of the required buffers of downstream operations only becomes known after the
    # sort, we require a host sync to read back the total intersection count. We mask this by
    # overlapping the readback with the tile binning and depth min-max pre-pass, which are
    # count-independent and can execute while the host waits for the count.
    pending = begin_count_read(cum_tiles_hit, total - 1, device)  # Enqueue the readback

    isect_dtype = wp.int32 if _32bit_packed else wp.int64
    scratch = cached_scratch(device, (B, n, n_tiles), 2, bins_len, isect_dtype)

    tile_bins = scratch["tile_bins"]
    tile_bins.zero_()  # The cache does not reset tile_bins for each frame, so zero out stale values
    if _32bit_packed:
        # Compute per-image [dmin, dmax] for the depth quantization device-side with no host sync.
        # Launched before the readback await to fill up the bubble we'd otherwise have on the GPU.
        # The cache again does not reset depth_mm for each frame, so we have to initialize it
        depth_mm = scratch["depth_mm"]
        cached_launch(seed_minmax, B, [depth_mm], device)
        dim = (total + MINMAX_CHUNK - 1) // MINMAX_CHUNK
        cached_launch(depth_minmax, dim, [depths, radii, total, n, depth_mm], device)

    n_intersects = fetch_count_read(pending)  # Force the readback to complete
    # Grow the sort buffers to the frame's count
    scratch = cached_scratch(
        device, (B, n, n_tiles), max(2 * n_intersects, 2), bins_len, isect_dtype
    )
    isect_ids = scratch["isect_ids"]
    gaussian_ids = scratch["gaussian_ids"]
    assert isect_ids is not None and gaussian_ids is not None

    if n_intersects == 0:
        return gaussian_ids, tile_bins, 0, tile_bounds_x, n_tiles

    # We can pass the full scratch arrays and don't need per-frame slicing or zeroing because all
    # accesses stay inside the valid range. The 32 and 64 bit paths take different kernel arguments,
    # the 32 bit path additionally quantizes depth into the packed key.
    if _32bit_packed:
        cached_launch(
            map_intersects_32bit,
            total,
            [
                xys,
                depths,
                radii,
                conics,
                map_opacities,
                cum_tiles_hit,
                depth_mm,
                n,
                opac_mod,
                tile_n_bits,
                depth_bits,
                tile_bounds_x,
                tile_bounds_y,
                isect_ids,
                gaussian_ids,
            ],
            device,
        )
        wp.utils.radix_sort_pairs(isect_ids, gaussian_ids, n_intersects)
        args = [n_intersects, isect_ids, n_tiles, tile_n_bits, depth_bits, tile_bins]
        cached_launch(tile_bin_edges_32bit, n_intersects, args, device)
    else:
        cached_launch(
            map_intersects_64bit,
            total,
            [
                xys,
                depths.view(wp.int32),  # depth is not normalized by the min-max pre-pass
                radii,
                conics,
                map_opacities,
                cum_tiles_hit,
                n,
                opac_mod,
                tile_n_bits,
                tile_bounds_x,
                tile_bounds_y,
                isect_ids,
                gaussian_ids,
            ],
            device,
        )
        wp.utils.radix_sort_pairs(isect_ids, gaussian_ids, n_intersects)
        args = [n_intersects, isect_ids, n_tiles, tile_n_bits, tile_bins]
        cached_launch(tile_bin_edges_64bit, n_intersects, args, device)
    return gaussian_ids, tile_bins, n_intersects, tile_bounds_x, n_tiles


def _use_32bit_keys(depth_bits: int) -> bool:
    """Check if we can use 32-bit keys for the sort, given the number of bits available for depth.

    Note:
        Do not inline this function. It is used in the test suite to verify the packed key paths.
    """
    return depth_bits >= 16
