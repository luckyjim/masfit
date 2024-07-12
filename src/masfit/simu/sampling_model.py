"""


"""

import numpy as np
from .model_src import moffat_model_2d, random_moffat


###################  MESH


def reg_mesh_center_pixel(s_patch):
    x = np.arange(s_patch).astype(np.float32) - s_patch // 2
    xv, yv = np.meshgrid(x, x)
    print(xv)
    return xv, yv


def reg_mesh_center_pixel_2d(s_patch):
    x = np.arange(s_patch).astype(np.float32) - s_patch // 2
    xv, yv = np.meshgrid(x, x)
    xy = np.array([xv, yv]) + 0.5
    xy = np.moveaxis(xy, 0, 2)
    return xy


def xy_mesh_center_pixel(s_patch):
    x = np.arange(s_patch).astype(np.float32) - s_patch // 2
    xv, yv = np.meshgrid(x, x)
    xy = np.empty((s_patch**2, 2), dtype=np.float32)
    xy[:, 0] = xv.ravel()
    xy[:, 1] = yv.ravel()
    return xy



###################  SIMULATE PATCH


def sampling_model_2d(m_moffat, s_patch):
    xv, yv = reg_mesh_center_pixel(s_patch)
    out = np.array([xv, yv])
    out = np.moveaxis(out, 0, 2)
    print("out ", out.shape)
    print(out[1, 2, 0], out[1, 2, 1])
    print(m_moffat(out))


def sampling_array_model(coef_mof, s_patch):
    nb_m = coef_mof.shape[0]
    xy = reg_mesh_center_pixel_2d(s_patch)
    sam = np.empty((nb_m, s_patch, s_patch), dtype=np.float32)
    for idx in range(nb_m):
        base = coef_mof[idx, 0]
        amp = coef_mof[idx, 1]
        x_0 = coef_mof[idx, 2]
        y_0 = coef_mof[idx, 3]
        sig_x = coef_mof[idx, 4]
        sig_y = coef_mof[idx, 5]
        b_pow = coef_mof[idx, 6]
        offset = np.floor(np.array([x_0, y_0], dtype=np.float32))
        print(offset)
        n_xy = xy + offset
        sam[idx] = moffat_model_2d(n_xy, base, amp, x_0, y_0, sig_x, sig_y, b_pow)
    return sam


###################  SIMULATE IMAGE


def patch_to_image(coef_mof, nb_pix, s_patch):
    nb_m = coef_mof.shape[0]
    sam = sampling_array_model(coef_mof, s_patch)
    sky_ima = np.zeros((nb_pix, nb_pix), dtype=np.float32)
    half_patch = s_patch // 2
    for idx in range(nb_m):
        x_0 = coef_mof[idx, 2]
        y_0 = coef_mof[idx, 3]
        print(x_0, y_0)
        corner_lb = np.floor(np.array([x_0, y_0])).astype(np.int64) - half_patch
        lbx = corner_lb[0]
        lby = corner_lb[1]
        print(sam[idx].shape, half_patch)
        print(lbx, lbx + s_patch, lby, lby + s_patch)
        sky_ima[lbx : lbx + s_patch, lby : lby + s_patch] += sam[idx]
    return sky_ima, sam


def simulate_image(nb_src=120, nb_pix=1024, s_patch=51):
    coef = random_moffat(nb_src, s_patch, nb_pix)
    sky_ima, sam = patch_to_image(coef, nb_pix, s_patch)
    return sky_ima, sam, coef
