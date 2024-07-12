from masfit.simu.sampling_model import *
import matplotlib.pyplot as plt
import pprint

import numpy as np

nb = 5
xy = np.random.uniform(-2, 2, 2 * nb).astype(np.float32).reshape(nb, 2)
xy[0] = [10, 20]
# print(xy)


def test_moffat_1():
    z_mof = moffat_model(xy, 1, 2, 10, 20, 1, 1, 1)
    print(z_mof)


def test_moffat_2():
    z_mof = moffat_test(xy)
    print(z_mof)


def test_moffat_3():
    nb_pix = 16
    xy = np.random.uniform(-2, 2, nb_pix * nb_pix * 2).astype(np.float32).reshape(nb_pix, nb_pix, 2)
    print(xy.shape)
    z_mof = moffat_test_2d(xy)
    print("out shape: ", z_mof.shape)
    pprint.pprint(z_mof)


def test_center_pixel(nb):
    xv, yv = reg_mesh_center_pixel(nb)
    print(xv)
    print(yv)


def test_sampling_2d(nb):
    sampling_moffat_2d(5)


def simu_moffat():
    nb_src = 1
    nb_pix = 64
    s_patch = 7
    coef = random_moffat(nb_src, nb_pix)
    sam = sampling_array_model(coef, s_patch)
    plt.imshow(sam[0])


def test_simulation_image():
    nb_src = 120
    nb_pix = 1024
    s_patch = 51
    coef = random_moffat(nb_src, s_patch, nb_pix)
    sky_ima, sam = patch_to_image(coef, nb_pix, s_patch)
    plt.imshow(sky_ima)


if __name__ == "__main__":
    # test_moffat_1()
    # test_moffat_2()
    # test_moffat_3()
    # test_center_pixel(7)
    # sampling_moffat_2d(moffat_test_2d, 5)
    # simu_moffat()
    test_simulation_image()
    plt.show()
