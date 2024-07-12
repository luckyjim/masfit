import numpy as np
from functools import partial

###################  MOFFAT MODEL


def moffat_model_flat(xy, base, amp, x_0, y_0, sig_x, sig_y, b_pow):
    """
    in :
     * xy : float(n,2)

    return is "flat":
     * z : float (n,)
    """
    center = np.array([[x_0, y_0]], dtype=np.float32)
    pos_l = xy - center
    print("xy")
    print(xy.shape)
    print(center.shape)
    print(pos_l.shape)
    print(pos_l)
    var_xy = np.array([[sig_x**2, sig_y**2]], dtype=np.float32)
    dist = np.sum(pos_l * pos_l / var_xy, axis=1)
    z_mo = base + amp * np.power(1 + dist, -b_pow)
    return z_mo


def moffat_model_2d(xy, base, amp, x_0, y_0, sig_x, sig_y, b_pow):
    """
    in :
     * xy : float(n,n,2)

    return is 2d:
     * z : float (n,n)
    """
    # print("moffat_model_2d xy", xy.shape)
    # print(x_0, y_0)
    center = np.array([[x_0, y_0]], dtype=np.float32)
    pos_l = xy - center
    var_xy = np.array([[sig_x**2, sig_y**2]], dtype=np.float32)
    dist = np.sum(pos_l * pos_l / var_xy, axis=2)
    #print("dist: ", dist.shape)
    z_mo = base + amp * np.power(1 + dist, -b_pow)
    #print(z_mo.shape)
    return z_mo


def random_moffat(nb_src, s_patch, nb_pix=512):
    """
    0: base,
    1: amp,
    2: x_0,
    3: y_0,
    4: sig_x,
    5: sig_y,
    6: b_pow

    """
    print(nb_src)
    coef_mof = np.zeros((nb_src, 7), dtype=np.float32)
    coef_mof[:, 0] = np.random.uniform(1, 10, nb_src)
    coef_mof[:, 1] = np.random.uniform(20, 1000, nb_src)
    coef_mof[:, 2] = np.random.uniform(s_patch, nb_pix - s_patch, nb_src)
    coef_mof[:, 3] = np.random.uniform(s_patch, nb_pix - s_patch, nb_src)
    coef_mof[:, 4] = np.random.uniform(1, 15, nb_src)
    # coef_mof[:, 5] = np.random.uniform(1, 10, nb_src)
    coef_mof[:, 5] = coef_mof[:, 4].copy()
    coef_mof[:, 6] = np.random.uniform(2, 5, nb_src)
    return coef_mof


moffat_test = partial(moffat_model_flat, base=1, amp=10, x_0=10, y_0=20, sig_x=2, sig_y=2, b_pow=2)
moffat_test_2d = partial(moffat_model_2d, base=1, amp=10, x_0=10, y_0=20, sig_x=2, sig_y=2, b_pow=2)
