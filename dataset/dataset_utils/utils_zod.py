import torch
import numpy as np
import torch.nn.functional as F



def rotate_image_torch(image, angle_deg, center=None):
    """
    Use instead of cv2.getRotationMatrix2D(), cv2.warpAffine()
    """
    # image: H x W x C, np.float32 in [0,1] or [0,255]
    h, w, c = image.shape
    img_t = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()  # 1 x C x H x W

    # angle in radians
    angle = np.deg2rad(angle_deg)
    cos, sin = np.cos(angle), np.sin(angle)

    # center of rotation in normalized coordinates [-1,1]
    if center is None:
        cx, cy = w / 2.0, h / 2.0
    else:
        cx, cy = center

    # translate so that center -> origin, then rotate, then translate back
    # Build 2x3 matrix in normalized coords
    # First compute translation offsets in pixel space
    tx = (2.0 * cx / (w - 1)) - 1.0
    ty = (2.0 * cy / (h - 1)) - 1.0

    # affine is applied in normalized coordinates; easiest is:
    #   shift to origin, rotate, shift back
    theta = torch.tensor([[
        [ cos, -sin, (1 - cos) * tx + sin * ty],
        [ sin,  cos, (1 - cos) * ty - sin * tx],
    ]], dtype=torch.float32)

    grid = F.affine_grid(theta, img_t.size(), align_corners=False)
    img_rot = F.grid_sample(img_t, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    out = img_rot.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return out.astype(image.dtype)

def resize_bilinear(image, new_h, new_w):
    """
    Resize function using only numpy to avoid cv2 in __get_item__()
    """
    # image: H x W x C (np.float32 or uint8)
    h, w = image.shape[:2]
    if h == new_h and w == new_w:
        return image

    # normalized coordinate grid in output
    ys = np.linspace(0, h - 1, new_h)
    xs = np.linspace(0, w - 1, new_w)
    xs, ys = np.meshgrid(xs, ys)

    x0 = np.floor(xs).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y0 = np.floor(ys).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, h - 1)

    wa = (x1 - xs) * (y1 - ys)
    wb = (xs - x0) * (y1 - ys)
    wc = (x1 - xs) * (ys - y0)
    wd = (xs - x0) * (ys - y0)

    wa = wa[..., None]
    wb = wb[..., None]
    wc = wc[..., None]
    wd = wd[..., None]

    Ia = image[y0, x0]
    Ib = image[y0, x1]
    Ic = image[y1, x0]
    Id = image[y1, x1]

    out = wa * Ia + wb * Ib + wc * Ic + wd * Id
    return out.astype(image.dtype)

def approximate_meridian_convergence(lat_deg, lon_deg, central_meridian_deg):
    """
    Approximates the meridian convergence (in degrees) given a point's latitude and longitude,
    and the central meridian of the map projection. (from Zimin in open_visloc_benchmark)

    Args:
        lat_deg (float): Latitude in degrees.
        lon_deg (float): Longitude in degrees.
        central_meridian_deg (float): Central meridian longitude in degrees.

    Returns:
        float: Meridian convergence in degrees.
    """
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    cm_rad = np.radians(central_meridian_deg)
    gamma_rad = np.arctan(np.tan(lon_rad - cm_rad) * np.sin(lat_rad))
    return np.degrees(gamma_rad)

def voxel_downsample(points, voxel_size):
    """
    Downsample point cloud without using o3d. 
    """
    # points: N x 3 (np.float32)
    if points.shape[0] == 0:
        return points

    # Compute voxel indices
    coords_min = points.min(axis=0)
    voxel_indices = np.floor((points - coords_min) / voxel_size).astype(np.int64)  # N x 3

    # Map each voxel index to a list of points; do this via dict of sums/counts
    voxel_dict = {}
    for idx, v in enumerate(voxel_indices):
        key = (v[0], v[1], v[2])
        if key in voxel_dict:
            voxel_dict[key][0] += points[idx]
            voxel_dict[key][1] += 1
        else:
            voxel_dict[key] = [points[idx].copy(), 1]

    # Compute centroids
    out = np.empty((len(voxel_dict), 3), dtype=points.dtype)
    for i, (key, (sum_pts, count)) in enumerate(voxel_dict.items()):
        out[i] = sum_pts / count

    return out

def create_bev_pointmap(points, 
               xy_range=50.0, 
               grid_size=480,
               z_min_default=-5.0,
               z_max_default=30.0):
    """
    points: [N, 4] array of x,y,z,intensity
    Assumes points already cropped into [-xy_range, xy_range] in both X and Y.
    
    Returns:
        bev: [5, 480, 480] tensor
    """

    # grid resolution in meters per pixel
    res = xy_range / grid_size  # 50 / 480 ~= 0.1041666667

    # Allocate grids
    C = 6
    H = W = grid_size

    count          = np.zeros((H, W), dtype=np.float32)
    z_max          = np.full((H, W), z_min_default, dtype=np.float32)
    z_min          = np.full((H, W), z_max_default, dtype=np.float32)
    z_sum          = np.zeros((H, W), dtype=np.float32)
    z_sq_sum       = np.zeros((H, W), dtype=np.float32)

    # Extract columns
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Convert to grid coordinates
    ix = ((x + xy_range) / res).astype(np.int32)
    iy = ((y + xy_range) / res).astype(np.int32)

    # Keep only in-bounds points
    mask = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    ix = ix[mask]
    iy = iy[mask]
    z = z[mask]

    # Aggregate
    for gx, gy, gz in zip(ix, iy, z):
        count[gy, gx] += 1
        z_max[gy, gx] = max(z_max[gy, gx], gz)
        z_min[gy, gx] = min(z_min[gy, gx], gz)
        z_sum[gy, gx] += gz
        z_sq_sum[gy, gx] += gz * gz

    # Compute means and std
    nonzero = count > 0

    z_mean = np.zeros_like(z_sum)
    z_std = np.zeros_like(z_sum)

    z_mean[nonzero] = z_sum[nonzero] / count[nonzero]

    var = (z_sq_sum - count * z_mean**2)
    var[nonzero] /= np.maximum(count[nonzero] - 1, 1)
    z_std[nonzero] = np.sqrt(np.maximum(var[nonzero], 0))

    vertical_extent = (z_max - z_min)

    # Stack into BEV
    bev = np.stack([
        count,
        z_max,
        z_mean,
        z_std,
        vertical_extent
    ], axis=0)

    return bev

