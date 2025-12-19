
import numpy as np

from hikerverseuniverse.celestials.star_field import StarField
from hikerverseuniverse.sensor_physics.optical_sensor_implementation import OpticalSensorImpl
from hikerverseuniverse.universe_api import all_celestials_within_distance
from hikerverseuniverse.utils.math_utils import gaussian_psf

PC_TO_M = 3.085677581e16  # meters per parsec



# --- Example Usage ---
if __name__ == "__main__":
    stars_ = all_celestials_within_distance(coordinates=[0, 0, 0, ], distance=60)

    positions = []
    luminosities = []
    temperatures = []
    radii = []
    for st in stars_:
        positions.append([(float(st.x)), (float(st.y)), (float(st.z))])
        luminosities.append(float(st.luminosity))
        temperatures.append(float(st.temperature))
        radii.append(float(st.radius))

    positions = np.array(positions) * PC_TO_M
    luminosities = np.array(luminosities)
    temperatures = np.array(temperatures)
    radii = np.array(radii)


    star_field = StarField(positions, temperatures, luminosities, radii)

    # Define telescope
    #telescope = OpticalSensorImpl(fov_deg=45, resolution=(512, 512))
    #telescope.set_star_file(star_file=star_field)

    cam_pos = np.array([0, 0, 0])  # 4.84814e-7
    cam_dir = np.array([0, 0, -1])
    up_hint = np.array([0, 1, 0])
    threshold = 5e-18
    exposure = 100  # Adjust this value to brighten dim stars
    saturation_limit = 1e-10  # Maximum pixel intensity
    blooming_factor = 0.2 # Spread of overexposed light
    image, intensity_img = OpticalSensorImpl.render(psf=gaussian_psf(3, 1),
                                                    star_field=star_field,
                                                    band_center_m = 550e-9,
                                                    aperture_diameter=1,
                                                    fov_deg=45,
                                                    resolution=(512, 512),
                                                    telescope_position=cam_pos,
                                                    camera_direction=cam_dir,
                                                    up_hint=up_hint, threshold=threshold,
                                                    exposure=exposure, saturation_limit=saturation_limit,
                                                    blooming_factor=blooming_factor, log_scale=False,
                                                    gain=1)





    # Save or display the image
    from PIL import Image

    img = (image / image.max() * 255).astype(np.uint8)
    Image.fromarray(img).save("telescope_view.png")

    img_grey = (intensity_img / intensity_img.max() * 65535).astype(np.uint16)
    Image.fromarray(img_grey).save("telescope_view_intensity.png")

    vecs = OpticalSensorImpl.calculate_star_vectors(star_field=star_field,
                                                    cam_pos=cam_pos,
                                                    cam_dir=cam_dir,
                                                    up_hint=up_hint)
    for i, vec in enumerate(vecs):
        print(f"Star {i}: X: {vec[0]:.22f}, Y: {vec[1]:.22f}, Z: {vec[2]:.22f}")


    star_locations = OpticalSensorImpl.estimate_star_locations(fov_deg=45,
                                                               resolution=(512, 512),
                                                               image=image,
                                                               threshold=threshold,
                                                               cam_pos=cam_pos, cam_dir=cam_dir, up_hint=up_hint)
    for star_loc in star_locations:
        print(f"X: {star_loc[0]:.2f}, Y: {star_loc[1]:.2f}, Temp: {star_loc[2]:.2f} K, Intensity: {star_loc[3]:.2e}")


    # Detect stars and compute camera->star direction vectors
    results = OpticalSensorImpl.scan_image_for_star_vectors(
        resolution=(512, 512),
        fov_deg=45,
        image=image,
        threshold=0.1e-15,            # threshold on brightness map (tune for your images)
        cam_pos=cam_pos,
        cam_dir=cam_dir,
        up_hint=up_hint,
        depth_scale=1000.0        # simple intensity->distance scaling for demo
    )

    # Print results
    for i, (dir_vec, est_pos, (px, py), intensity) in enumerate(results):
        print(f"Star {i}: pixel=({px},{py}), intensity={intensity:.3e}")
        print(f"  direction (world, unit) = {dir_vec}")
        # if est_pos is not None:
        #     print(f"  estimated position = {est_pos}")
        # else:
        #     print("  estimated position = None")

    OpticalSensorImpl.show_interactive_starfield(star_field=star_field, fov_deg=45, cam_pos=cam_pos, cam_dir=cam_dir, up_hint=up_hint)
