

import numpy as np

from hikerverseuniverse.universe_api import all_celestials_within_distance
from sfs_lib import (
    tone_map,
    tone_map_to_u16,
    render_stars_rgb_temperature, reconstruct_star_field_from_image_per_temp,
    apply_time_integration, render_stars_rgb_temperature2,
)

if __name__ == "__main__":
    stars_ = all_celestials_within_distance(coordinates=[0, 0, 0, ], distance=50)

    stars_pos = []
    stars_lum = []
    stars_temp_k = []
    for st in stars_:
        stars_pos.append([(float(st.x)), (float(st.y)), (float(st.z))])
        stars_lum.append(float(st.luminosity))
        stars_temp_k.append(float(st.temperature))

    stars_pos = np.array(stars_pos)
    stars_lum = np.array(stars_lum)
    stars_temp_k = np.array(stars_temp_k)

    stars_temp_k[0] = 4000

    # 1) Define a small synthetic star field
    stars_pos2 = np.array([
        [0.0, 0.0, 0.0],    # roughly in the center
        [0.2, 0.0, 0.1],    # right
        [-0.1, 0.1, 0.2],   # left/up
    ], dtype=np.float64)

    stars_lum2 = np.array([
        3.828e27,   # bright
        3.828e26,   # medium
        3.828e25,   # faint
    ], dtype=np.float64)

    stars_spec = np.array(["G", "K", "M"], dtype=object)

    stars_temp_k2 = np.array([6000.0, 8000.0, 3500.0], dtype=np.float64)

    cam_pos = np.array([0.0, 0.0, -60], dtype=np.float64)
    cam_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    img_width = 512
    img_height = 512
    fov_y_deg = 90.0
    det_threshold_w_m2 = 1e-45
    det_threshold_energy = 1e-45

    # 2) Sanity-check that all stars are in front of the camera
    for star in stars_pos:
        if np.linalg.norm(star - cam_pos) < 1e-6:
            raise ValueError("Star position coincides with camera position.")
        dir_to_star = star - cam_pos
        dir_to_star /= np.linalg.norm(dir_to_star)
        #if np.dot(dir_to_star, cam_dir) <= 0:
        #    raise ValueError("All stars must be in front of the camera.")

    # 3) Render physical RGB flux image (linear W/m^2 per channel)
    flux_img_rgb = render_stars_rgb_temperature(
        stars_pos_pc=stars_pos,
        stars_lum_w=stars_lum,
        stars_temp_k=stars_temp_k,
        cam_pos_pc=cam_pos,
        cam_dir=cam_dir,
        up_hint=up_hint,
        fov_y_deg=fov_y_deg,
        img_width=img_width,
        img_height=img_height,
        det_threshold_w_m2=det_threshold_w_m2,
    )  # (H, W, 3)

    from sfs_lib import gaussian_kernel

    psf_kernel = gaussian_kernel(radius_px=3, sigma_px=.1)

    flux_img_rgb2 = render_stars_rgb_temperature2(
        stars_pos_pc=stars_pos,
        stars_lum_w=stars_lum,
        stars_temp_k=stars_temp_k,
        cam_pos_pc=cam_pos,
        cam_dir=cam_dir,
        up_hint=up_hint,
        fov_y_deg=fov_y_deg,
        img_width=img_width,
        img_height=img_height,
        det_threshold_w_m2=det_threshold_w_m2,
        psf_kernel=psf_kernel,
    )




    t_exp_s = 1  # 10-second exposure vs e.g. 1-second exposure

    energy_img_rgb = apply_time_integration(
        flux_img_rgb=flux_img_rgb,
        t_exp_s=t_exp_s,
        add_noise=False,
        read_noise_std=1e-15,
    )

    # 4) Build total energy image for reconstruction [J/m^2]
    energy_img_gray = energy_img_rgb.sum(axis=-1)  # (H, W)

    # 5) Build normalized RGB from energy for temperature inference
    rgb_for_temp = energy_img_rgb.copy().astype(np.float64)
    total_energy = rgb_for_temp.sum(axis=-1, keepdims=True)
    total_energy[total_energy == 0.0] = 1.0
    rgb_for_temp /= total_energy

    # 6) Tone map from energy to displayable images
    exposure = 1e22  # adjusted since values are larger than flux by t_exp_s
    gamma = 0.2
    method = "log"

    img_u8_rgb = np.zeros_like(energy_img_rgb, dtype=np.uint8)
    img_u16_rgb = np.zeros_like(energy_img_rgb, dtype=np.uint16)
    for c in range(3):
        img_u8_rgb[..., c] = tone_map(
            flux_image=energy_img_rgb[..., c],
            exposure=exposure,
            gamma=gamma,
            method=method,
        )
        img_u16_rgb[..., c] = tone_map_to_u16(
            flux_image=energy_img_rgb[..., c],
            exposure=exposure,
            gamma=gamma,
            method=method,
        )

    # 7) Reconstruct using energy image; function converts back to flux internally
    # star_pos_pc, star_lum_w_rec, star_temp_rec = reconstruct_star_field_from_image_per_temp(
    #     energy_img=energy_img_gray,
    #     rgb_energy_img=rgb_for_temp,
    #     cam_pos_pc=cam_pos,
    #     cam_dir=cam_dir,
    #     up_hint=up_hint,
    #     fov_y_deg=fov_y_deg,
    #     det_threshold_energy=det_threshold_energy,
    #     t_exp_s=t_exp_s,
    # )

    energy_gray = energy_img_rgb.sum(axis=-1)
    img_u16_gray = tone_map_to_u16(
        flux_image=energy_gray,
        exposure=exposure,
        gamma=gamma,
        method=method,
    )

    # 8) Print results
    # print("Input stars:")
    # for i in range(stars_pos.shape[0]):
    #     print(f"Star {i}: pos={stars_pos[i]}, L={stars_lum[i]:.3e}, spec={stars_spec[i]}")
    #
    # print("\nReconstructed stars:")
    # for i in range(star_pos_pc.shape[0]):
    #     print(f"Star {i}:")
    #     print(f"  Position (pc): {star_pos_pc[i]}")
    #     print(f"  Luminosity (W): {star_lum_w_rec[i]:.3e}")
    #     print(f"  Temperature (K): {star_temp_rec[i]:.1f}")

    # 9) Save rendered images

    from PIL import Image

    img_pil_8 = Image.fromarray(img_u8_rgb, mode="RGB")
    img_pil_8.save("star_field_rgb_8bit.png")

    gray_u16 = tone_map_to_u16(
        flux_image=energy_img_gray,
        exposure=exposure,
        gamma=gamma,
        method=method,
    )
    img_pil_16 = Image.fromarray(gray_u16)
    img_pil_16.save("star_field_gray_16bit.png")

    Image.fromarray(img_u8_rgb, mode="RGB").save("star_field_energy_rgb_8bit.png")
    Image.fromarray(img_u16_gray).save("star_field_energy_gray_16bit.png")

