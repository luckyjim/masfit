"""
Created on 12 juil. 2024

@author: jcolley
"""
import pprint

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from masfit.simu.model_src import moffat_model_2d
from masfit.simu.sampling_model import reg_mesh_center_pixel_2d

def moffat_model_2d_pa(xy, pars):
    base = pars[0]
    amp = pars[1]
    x_0 = pars[2]
    y_0 = pars[3]
    sig_x = pars[4]
    b_pow = pars[5]
    z_val = moffat_model_2d(xy, base, amp, x_0, y_0, sig_x, sig_x, b_pow)
    return z_val

def loss_moffat(pars, patch, xy):
    base = pars[0]
    amp = pars[1]
    x_0 = pars[2]
    y_0 = pars[3]
    sig_x = pars[4]
    b_pow = pars[5]
    # patch, xy = data
    z_val = moffat_model_2d(xy, base**2, amp, x_0, y_0, sig_x**2, sig_x**2, b_pow)
    #plt.figure()
    #plt.imshow(z_val)
    dif = z_val - patch
    loss = np.sum(dif * dif)
    #print(pars)
    print(loss)
    return loss


def fit_moffat_model(patch):
    s_x, s_y = patch.shape[0], patch.shape[1]
    assert s_x == s_y
    base = np.sqrt(patch[0, 0])
    amp = np.max(patch)/10
    # x_0, y_0 = np.unravel_index(patch.argmax(), patch.shape)
    x_0, y_0 = 0.0, 0.0
    sig_x = 2
    # sig_y = 2
    b_pow = 2
    pars_0 = np.array([base, amp, x_0, y_0, sig_x, b_pow],dtype=np.float64)
    xy = reg_mesh_center_pixel_2d(s_x)
    assert xy.ndim == 3
    pprint.pprint(xy)
    data = (patch.astype(np.float64), xy)
    res = minimize(loss_moffat, pars_0, args=data, method="BFGS", options={"eps": 1e-3,"disp": True})
    pars_est = res.x.copy()
    pars_est[0] *= pars_est[0]
    pars_est[4] *= pars_est[4]
    print(pars_est)
    z_est = moffat_model_2d_pa(xy,pars_est)
    plt.figure()
    plt.imshow(z_est)  
