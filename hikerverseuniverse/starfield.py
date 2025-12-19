# python
import numpy as np
from PIL import Image

from hikerverseuniverse.image_tester import render_starfield, to_uint8

star_locs = [(1.63e-05, 0.0, 0.0), (1.53958064, 1.17833026, 3.75297394), (1.61429006, 1.34955198, 3.77075724),
             (1.61436178, 1.34991384, 3.7705975), (0.05663598, 5.92215838, 0.48614098),
             (7.44196386, 2.11682884, 0.95210582),
             (6.51348, 1.6440343, 4.87534304), (7.40182674, 3.4131222, 2.64047938), (7.40225706, 3.41332106, 2.6406326),
             (1.61149298, 8.07414306, 2.4726611), (1.61167554, 8.07441364, 2.4726122),
             (1.90997858, 8.64700004, 3.91240098),
             (1.63e-05, 0.0, 0.0), (1.53958064, 1.17833026, 3.75297394), (1.61429006, 1.34955198, 3.77075724),
             (1.61436178, 1.34991384, 3.7705975), (0.05663598, 5.92215838, 0.48614098),
             (7.44196386, 2.11682884, 0.95210582),
             (6.51348, 1.6440343, 4.87534304), (7.40182674, 3.4131222, 2.64047938), (7.40225706, 3.41332106, 2.6406326),
             (1.61149298, 8.07414306, 2.4726611), (1.61167554, 8.07441364, 2.4726122),
             (1.90997858, 8.64700004, 3.91240098),
             (1.63e-05, 0.0, 0.0), (1.53958064, 1.17833026, 3.75297394), (1.61429006, 1.34955198, 3.77075724),
             (1.61436178, 1.34991384, 3.7705975), (0.05663598, 5.92215838, 0.48614098),
             (7.44196386, 2.11682884, 0.95210582),
             (6.51348, 1.6440343, 4.87534304), (7.40182674, 3.4131222, 2.64047938), (7.40225706, 3.41332106, 2.6406326),
             (1.61149298, 8.07414306, 2.4726611), (1.61167554, 8.07441364, 2.4726122),
             (1.90997858, 8.64700004, 3.91240098)]

star_lums = [10047.021943573667, 0.579745452583711, 4430.342999135716, 15503.771208254617, 4.446695089097862,
             0.2085630411265554, 57.28452955046373, 0.566027264412932, 0.5021544906013337, 229317.58042370147,
             25.611637764519507, 5.4908218192357, 10047.021943573667, 0.579745452583711, 4430.342999135716,
             15503.771208254617, 4.446695089097862, 0.2085630411265554, 57.28452955046373, 0.566027264412932,
             0.5021544906013337, 229317.58042370147, 25.611637764519507, 5.4908218192357, 10047.021943573667,
             0.579745452583711, 4430.342999135716, 15503.771208254617, 4.446695089097862, 0.2085630411265554,
             57.28452955046373, 0.566027264412932, 0.5021544906013337, 229317.58042370147, 25.611637764519507,
             5.4908218192357]


def prepare_and_render(star_locs, star_lums, out_path='starfield_clean.png'):
    stars = np.asarray(star_locs, dtype=np.float64)
    lums = np.asarray(star_lums, dtype=np.float64)

    # align lengths
    n = min(len(stars), len(lums))
    stars = stars[:n]
    lums = lums[:n]

    # remove exact duplicate positions (preserve first-occurrence order)
    _, first_idx = np.unique(stars, axis=0, return_index=True)
    keep_idx = np.sort(first_idx)
    stars = stars[keep_idx]
    lums = lums[keep_idx]

    # ensure all points are in front of camera
    eps_z = 1e-3
    mask_front = stars[:, 2] <= eps_z
    if np.any(mask_front):
        stars[mask_front, 2] = eps_z

    # render and save (assumes render_starfield and to_uint8 are in scope)
    img = render_starfield(stars, lums, observer_pos=(0, 0, 0),
                           orientation_matrix=None, fov_deg=90.0,
                           image_size=(512, 512), psf_sigma_px=1.2)
    img8 = to_uint8(img, gamma=0.7)
    Image.fromarray(img8).save(out_path)
    return len(stars)


# Example call (use your existing `star_locs` and `star_lums`)
if __name__ == "__main__":
    count = prepare_and_render(star_locs, star_lums, out_path='starfield_clean.png')
    print(f"Rendered {count} unique stars to `starfield_clean.png`")
