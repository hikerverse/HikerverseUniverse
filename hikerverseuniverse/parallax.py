import numpy as np
from hikerverseuniverse.celestials.star_field import StarField
from hikerverseuniverse.library.constants import L0
from hikerverseuniverse.sensor_physics.optical_sensor_implementation import OpticalSensor

# Define the star field
positions = [[0, 0, 0], [125, 0, 0]]
temperatures = [16000, 6000]
luminosities = [L0, L0 / 10]

positions = np.array(positions)
luminosities = np.array(luminosities)
temperatures = np.array(temperatures)

star_field = StarField(positions, temperatures, luminosities)

# Define the telescope
telescope = OpticalSensor(fov_deg=90, resolution=(64, 64))
telescope.set_star_file(star_file=star_field)

# Define observation parameters
cam_dir = np.array([0, 0, -1])
up_hint = np.array([0, 1, 0])
threshold = 5e-14
exposure = 1
saturation_limit = 1e-12
blooming_factor = 4

# Observation 1: From the first location
cam_pos1 = np.array([0, 0, 500])
image1, image11 = telescope.render(cam_pos1, cam_dir, up_hint, threshold=threshold,
                             exposure=exposure, saturation_limit=saturation_limit,
                             blooming_factor=blooming_factor, log_scale=False, gain=1)

# Observation 2: From the second location (baseline shift)
cam_pos2 = np.array([100, 0, 500])  # Shifted along the X-axis
image2, image22 = telescope.render(cam_pos2, cam_dir, up_hint, threshold=threshold,
                             exposure=exposure, saturation_limit=saturation_limit,
                             blooming_factor=blooming_factor, log_scale=False, gain=1)

# Estimate star positions from both images
star_locations1 = telescope.scan_intensities(image11, 1e-20, cam_pos1, cam_dir, up_hint)
for star_loc in star_locations1:
    print(f"X: {star_loc[0]:.2f}, Y: {star_loc[1]:.2f}, Z: {star_loc[2]:.2f}, Temp: {star_loc[3]:.2f} K, Intensity: {star_loc[4]:.2e}")

star_locations2 = telescope.scan_intensities(image22, 1e-20, cam_pos2, cam_dir, up_hint)
for star_loc in star_locations2:
    print(f"X: {star_loc[0]:.2f}, Y: {star_loc[1]:.2f}, Z: {star_loc[2]:.2f}, Temp: {star_loc[3]:.2f} K, Intensity: {star_loc[4]:.2e}")

# Calculate actual positions using parallax
def calculate_positions(locations1, locations2, baseline):
    actual_positions = []
    for star1, star2 in zip(locations1, locations2):
        # Calculate parallax angle (in radians)
        parallax_angle = np.arctan2(abs(star2[0] - star1[0]), baseline)
        # Calculate distance using parallax formula: d = baseline / tan(parallax_angle)
        distance = baseline / np.tan(parallax_angle)
        # Calculate actual position
        actual_position = star1[:3] + (star2[:3] - star1[:3]) * distance / np.linalg.norm(star2[:3] - star1[:3])
        actual_positions.append(actual_position)
    return actual_positions

baseline = np.linalg.norm(cam_pos2 - cam_pos1)
actual_star_positions = calculate_positions(star_locations1, star_locations2, baseline)

# Print the calculated positions
for i, pos in enumerate(actual_star_positions):
    print(f"Star {i + 1}: X={pos[0]:.2f}, Y={pos[1]:.2f}, Z={pos[2]:.2f}")
