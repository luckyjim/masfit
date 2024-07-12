from masfit.star_fit import *
from masfit.simu.sampling_model import *
import matplotlib.pyplot as plt
import pprint

np.random.seed(10)


def test_fit_0():
    nb_src = 10
    nb_pix = 1024
    s_patch = 17
    coef = random_moffat(nb_src, s_patch, nb_pix)
    sky_ima, sam = patch_to_image(coef, nb_pix, s_patch)
    print("coef Moffat: ", coef[0])
    plt.figure()
    plt.imshow(sam[0])    
    fit_moffat_model(sam[0])
    print("coef Moffat: ", coef[0])
    

if __name__ == "__main__":
    test_fit_0()
    plt.show()
