import math
from math import pi, tan, radians
from typing import Tuple, Optional, List

import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve

from hikerverseuniverse.celestials.star import Star
from hikerverseuniverse.celestials.star_field import StarField
from hikerverseuniverse.library.constants import h, c
from hikerverseuniverse.sensor_physics.camera_utils import compute_camera_basis
from hikerverseuniverse.sfs_lib import rgb_to_temperature_new, temperature_to_rgb, PC_TO_M
from hikerverseuniverse.utils.math_utils import gaussian_psf, normalize

class OpticalSensorImpl:
    """
    Optical sensor (telescope + detector) model for game use.
    - aperture_diameter: m
    - throughput: fraction [0..1]
    - qe: quantum efficiency [0..1]
    - band_center_m, band_width_m: band
    - integration_time: s
    - background_photons_per_s: background rate (gamey)
    - read_noise_e: electrons RMS
    - fov_deg: field-of-view radius in degrees
    """

    def __init__(self, aperture_diameter: float = 1, throughput: float = 0.8, qe: float = 0.9,
                 band_center_m: float = 550e-9,
                 band_width_m: float = 100e-9,
                 resolution: Tuple[int, int] = (512, 512),
                 integration_time: float = 60.0,
                 background_photons_per_s: float = 0.0,
                 psf: Optional[List[List[float]]] = None,
                 read_noise_e: float = 5.0, fov_deg: float = 1.0):
        self.aperture_diameter = float(aperture_diameter)
        self.throughput = float(throughput)
        self.qe = float(qe)
        self.band_center_m = float(band_center_m)
        self.band_width_m = float(band_width_m)
        self.integration_time = float(integration_time)
        self.background_photons_per_s = float(background_photons_per_s)
        self.read_noise_e = float(read_noise_e)
        self.fov_deg = float(fov_deg)
        self.resolution = resolution  # (height, width)
        self.psf = psf if psf is not None else gaussian_psf(3, 1)

        self.star_field = None


    @staticmethod
    def _print_debug(title, img_16bit_grey, img_8bit_colour, debug=True):
        if debug:
            print(f"{title}")
            print(
                f"img_16bit_grey:  "
                f"Max value = {img_16bit_grey.max():<10.4e}   "
                f"Min value = {img_16bit_grey.min():<10.4e}   "
                f"Min value > 0 = {img_16bit_grey[img_16bit_grey > 0].min():<10.4e}   "
                f"Pixels > 0: {np.count_nonzero(img_16bit_grey):<10}")
            print(
                f"img_8bit_colour: "
                f"Max value = {img_8bit_colour.max():<10.4e}   "
                f"Min value = {img_8bit_colour.min():<10.4e}   "
                f"Min value > 0 = {img_8bit_colour[img_8bit_colour > 0].min():<10.4e} "
                f"Pixels > 0: {np.count_nonzero(img_8bit_colour) / 3:<10}\n")


    @staticmethod
    def take_image(psf, star_field, fov_deg, resolution, aperture_diameter, band_center_m,
                   telescope_position: np.ndarray, camera_direction: np.ndarray, up_hint: np.ndarray,
                   threshold: float,
                   exposure: float, saturation_limit: float, blooming_factor: float = 0.0,
                   log_scale: bool = False, min_flux=0.0, max_flux=None, gain=1) -> Tuple[np.ndarray, np.ndarray]:
        H, W = resolution
        fov_y_rad = radians(fov_deg)
        f = 1.0 / tan(fov_y_rad / 2.0)

        img_8bit_colour = np.zeros((H, W, 3), dtype=np.float64)
        img_16bit_grey = np.zeros((H, W), dtype=np.float64)

        # Camera basis
        forward, right, up = compute_camera_basis(camera_direction, up_hint)

        # Project stars onto the image plane
        star_data = OpticalSensorImpl.project_stars(star_field=star_field,
                                                    fov_deg=fov_deg,
                                                    resolution=resolution,
                                                    aperture_diameter=aperture_diameter,
                                                    band_center_m=band_center_m,
                                                    telescope_position=telescope_position,
                                                    forward=forward,
                                                    right=right,
                                                    up=up, f=f, H=H, W=W, threshold=threshold, exposure=exposure)

        for px, py, flux, color, radius, is_resolvable, radius_in_pixels in star_data:

            if is_resolvable:
                for dx in range(-radius_in_pixels, radius_in_pixels + 1):
                    for dy in range(-radius_in_pixels, radius_in_pixels + 1):
                        if dx ** 2 + dy ** 2 <= radius_in_pixels ** 2:  # Check if within circle
                            if 0 <= px + dx < W and 0 <= py + dy < H:
                                img_16bit_grey[px + dx, py + dy] += flux
                                img_8bit_colour[px + dx, py + dy, :] += flux * normalize(color)

            else:
                img_16bit_grey[px, py] += flux
                img_8bit_colour[px, py, :] += flux * normalize(color)

        # invert the image in the vertical axis
        img_8bit_colour = np.flipud(img_8bit_colour)
        img_16bit_grey = np.flipud(img_16bit_grey)

        debug = False

        OpticalSensorImpl._print_debug("raw", img_16bit_grey, img_8bit_colour, debug)

        # Apply gain to the image and grayscale arrays
        #img_8bit_colour, img_16bit_grey = self.apply_gain(img_8bit_colour, img_16bit_grey, gain=gain)  # Example gain factor


        # Apply PSF and simulate blooming
        img_8bit_colour, img_16bit_grey = OpticalSensorImpl.apply_psf_and_blooming(psf=psf,
                                                                                   img=img_8bit_colour,
                                                                                   grayscale_16bit=img_16bit_grey,
                                                                                   blooming_factor=blooming_factor,
                                                                                   saturation_limit=saturation_limit)
        OpticalSensorImpl._print_debug("After apply_psf_and_blooming", img_16bit_grey, img_8bit_colour, debug)

        img_8bit_colour, img_16bit_grey = OpticalSensorImpl.postprocess_image(image_data=img_8bit_colour,
                                                                              grayscale_16bit=img_16bit_grey,
                                                                              saturation_limit=saturation_limit,
                                                                              log_scale=log_scale,
                                                                              min_flux=min_flux,
                                                                              max_flux=max_flux)

        OpticalSensorImpl._print_debug("After postprocess_image", img_16bit_grey, img_8bit_colour, debug)



        return img_8bit_colour, img_16bit_grey

    @staticmethod
    def calculate_star_vectors(star_field, cam_pos: np.ndarray, cam_dir: np.ndarray, up_hint: np.ndarray) -> List[np.ndarray]:
        """
        Calculate vectors to the stars in the image relative to the camera's direction using higher precision.
        Returns vectors as numpy arrays with dtype np.longdouble.
        """
        if star_field is None:
            raise ValueError("Star field is not set. Please set the star field using `set_star_file`.")

        # Promote camera inputs to high precision
        cam_pos = np.asarray(cam_pos, dtype=np.longdouble)
        cam_dir = np.asarray(cam_dir, dtype=np.longdouble)
        up_hint = np.asarray(up_hint, dtype=np.longdouble)

        # Normalize camera basis vectors with high precision
        forward = normalize(cam_dir).astype(np.longdouble)
        right = normalize(np.cross(forward, up_hint)).astype(np.longdouble)
        up = np.cross(right, forward).astype(np.longdouble)

        pc_to_m = np.longdouble(PC_TO_M)

        star_vectors: List[np.ndarray] = []
        for pos in star_field.star_positions:
            p = np.asarray(pos, dtype=np.longdouble)

            # skip degenerate positions
            if np.all(p == np.longdouble(0)):
                continue

            # Vector from camera to star (convert star positions to meters if stored in parsecs)
            # Keep high precision throughout
            rel = (p - cam_pos)
            # Normalize the direction
            star_vector = normalize(rel).astype(np.longdouble)

            # Transform star vector to camera space with high precision
            star_vector_cam = np.array([
                np.dot(star_vector, right),
                np.dot(star_vector, up),
                np.dot(star_vector, forward)
            ], dtype=np.longdouble)

            star_vectors.append(star_vector_cam)

        return star_vectors

    @staticmethod
    def calculate_fov(sensor_width: float, focal_length: float) -> Tuple[float, float]:
        """
        Calculate the Field of View (FoV) of the optical telescope.

        :param sensor_width: Width of the sensor (in meters).
        :param focal_length: Focal length of the telescope (in meters).
        :return: FoV in radians and degrees as a tuple.
        """
        if focal_length <= 0:
            raise ValueError("Focal length must be greater than zero.")
        fov_rad = 2 * math.atan(sensor_width / (2 * focal_length))
        fov_deg = math.degrees(fov_rad)
        return fov_rad, fov_deg

    @staticmethod
    def apply_gain(img: np.ndarray, grayscale_16bit: np.ndarray, gain: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply gain to the image and intensify the flux values.

        :param img: The RGB image array.
        :param grayscale_16bit: The grayscale image array.
        :param gain: The gain factor to intensify the flux values.
        :return: Tuple of the modified RGB image and grayscale image.
        """
        img *= gain
        grayscale_16bit *= gain
        return img, grayscale_16bit

    @staticmethod
    def is_resolvable(fov_deg, resolution, band_center_m, aperture_diameter, radius: float) -> Tuple[bool, int]:
        """
        Check if the given radius is resolvable by the optical sensor.

        :param radius: Angular radius of the object in radians.
        :return: True if resolvable, False otherwise.
        """
        # Calculate the diffraction limit
        diffraction_limit = 1.22 * band_center_m / aperture_diameter

        # Calculate the pixel scale
        fov_rad = math.radians(fov_deg)
        pixel_scale = fov_rad / max(resolution)

        # Calculate the number of pixels the object spans
        pixel_diameter = 2 * radius / pixel_scale
        pixel_area = math.pi * (pixel_diameter / 2) ** 2
        radius_pixels = math.sqrt(pixel_area / math.pi)  # Approximate radius in pixels
        radius_in_pixels = int(radius_pixels)  # Convert to integer for pixel iteration

        if radius_in_pixels > 1:
            d = 3

        # Compare the radius to the diffraction limit and pixel scale
        return radius >= max(diffraction_limit, pixel_scale), radius_in_pixels

    @staticmethod
    def project_stars(star_field,
                      fov_deg,
                      aperture_diameter,
                      band_center_m,
                      resolution,
                      telescope_position: np.ndarray,
                      forward: np.ndarray,
                      right: np.ndarray, up: np.ndarray,
                      f: float, H: int, W: int, threshold: float, exposure: float) -> list:
        star_data = []
        for pos, L, T, R in zip(star_field.star_positions, star_field.star_luminosities,
                                star_field.star_temperatures, star_field.star_radii):
            v = pos - telescope_position
            d = np.linalg.norm(v)
            if d <= 0:
                continue

            dir_world = v / d
            if np.dot(dir_world, forward) <= 0.0:
                continue

            x_cam = np.dot(dir_world, right)
            y_cam = np.dot(dir_world, up)
            z_cam = np.dot(dir_world, forward)
            if z_cam <= 0:
                continue

            u = x_cam / (z_cam * f)
            v_ndc = y_cam / (z_cam * f)
            px = int((u + 1.0) * 0.5 * W)
            py = int((1.0 - (v_ndc + 1.0) * 0.5) * H)



            if 0 <= px < W and 0 <= py < H:
                flux = L / (4.0 * pi * d ** 2) * exposure
                color = temperature_to_rgb(T)
                radius = 2 * math.atan(R / d)
                is_resolvable, radius_in_pixels = OpticalSensorImpl.is_resolvable(fov_deg=fov_deg,
                                                                                  aperture_diameter=aperture_diameter,
                                                                                  band_center_m=band_center_m,
                                                                                  resolution=resolution,
                                                                                  radius=radius)
                if is_resolvable and radius_in_pixels > 1:
                    d = 2

                star_data.append((px, py, flux, color, radius, is_resolvable, radius_in_pixels))
        return star_data

    @staticmethod
    def estimate_star_locations(fov_deg, resolution, image: np.ndarray, threshold: float, cam_pos: np.ndarray, cam_dir: np.ndarray,
                                up_hint: np.ndarray) -> list:
        from scipy.ndimage import label, center_of_mass

        H, W = resolution
        fov_deg = radians(fov_deg)
        f = 1.0 / tan(fov_deg / 2.0)

        # Camera basis
        forward = normalize(cam_dir)
        right = normalize(np.cross(forward, up_hint))
        up = np.cross(right, forward)

        # Threshold the image to detect bright spots
        binary_image = (image.max(axis=2) > threshold).astype(int)

        # Label connected components (bright spots)
        labeled, num_features = label(binary_image)

        # Calculate centroids, estimate world coordinates, and temperatures
        star_data = []
        for i in range(1, num_features + 1):
            # Compute centroid (X, Y)
            centroid = center_of_mass(binary_image, labeled, i)
            x, y = centroid

            # Estimate depth (Z) based on intensity
            intensity = image[int(x), int(y), :].sum()

            # Convert pixel coordinates to NDC
            u = (x / W) * 2.0 - 1.0
            v = 1.0 - (y / H) * 2.0

            # Reconstruct direction in camera space
            dir_cam = normalize(np.array([u, v, -1.0 / f]))

            # Transform to world space
            dir_world = dir_cam[0] * right + dir_cam[1] * up + dir_cam[2] * forward

            # Calculate world coordinates
            world_pos = cam_pos + dir_world * 1.0

            # Estimate temperature from RGB color
            rgb = (image[int(x), int(y), :] / image.max()) * 255
            temperature = rgb_to_temperature_new(rgb)

            # Append world X and Y only (remove Z component), plus temperature and intensity
            star_data.append((float(world_pos[0]), float(world_pos[1]), temperature, intensity))

        return star_data

    @staticmethod
    def scan_intensities(fov_deg, resolution, grayscale_image: np.ndarray, threshold: float, cam_pos: np.ndarray, cam_dir: np.ndarray,
                         up_hint: np.ndarray) -> list:
        from scipy.ndimage import label, center_of_mass

        H, W = resolution
        fov_y_rad = math.radians(fov_deg)
        f = 1.0 / math.tan(fov_y_rad / 2.0)

        # Camera basis
        forward = normalize(cam_dir)
        right = normalize(np.cross(forward, up_hint))
        up = np.cross(right, forward)

        # Threshold the grayscale image to detect bright spots
        binary_image = (grayscale_image > threshold).astype(int)

        # Label connected components (bright spots)
        labeled, num_features = label(binary_image)

        # Calculate centroids and estimate world coordinates and flux intensities
        star_data = []
        for i in range(1, num_features + 1):
            # Compute centroid (X, Y)
            centroid = center_of_mass(binary_image, labeled, i)
            y, x = int(centroid[0]), int(centroid[1])  # Convert to integer pixel coordinates

            # Estimate depth (Z) based on intensity
            intensity = grayscale_image[y, x]
            z = 1.0 / np.sqrt(intensity) if intensity > 0 else float('inf')

            # Convert pixel coordinates to NDC
            u = (x / W) * 2.0 - 1.0
            v = 1.0 - (y / H) * 2.0

            # Reconstruct direction in camera space
            dir_cam = normalize(np.array([u, v, -1.0 / f]))

            # Transform to world space
            dir_world = dir_cam[0] * right + dir_cam[1] * up + dir_cam[2] * forward

            # Calculate world coordinates
            world_pos = cam_pos + dir_world * z

            # Append star data (world position and flux intensity)
            star_data.append((*world_pos, intensity))

        return star_data

    @staticmethod
    def scan_image_for_star_vectors(fov_deg, resolution, image: np.ndarray, threshold: float,
                                    cam_pos: np.ndarray, cam_dir: np.ndarray, up_hint: np.ndarray,
                                    depth_scale: float = 1.0) -> list:
        """
        Detect stars in `image` using a simple threshold, compute direction vectors from the camera
        position to each detected star in world coordinates, and return a list of tuples:
          (direction_unit_vector, estimated_world_position_or_None, (px, py), intensity)

        - image: HxWx3 RGB image (float or uint)
        - threshold: brightness threshold applied to max(R,G,B)
        - cam_pos, cam_dir, up_hint: camera world-space vectors
        - depth_scale: scalar used to convert intensity -> distance estimate (z = depth_scale / sqrt(intensity))
        """
        from scipy.ndimage import label, center_of_mass

        H, W = resolution
        fov_y_rad = math.radians(fov_deg)
        f = 1.0 / math.tan(fov_y_rad / 2.0)

        # Build camera basis (world-space)
        forward = normalize(cam_dir)
        right = normalize(np.cross(forward, up_hint))
        up = np.cross(right, forward)

        # Simple brightness map and threshold
        brightness = image.max(axis=2)
        binary = (brightness > threshold).astype(int)

        labeled, num_features = label(binary)
        results = []

        for i in range(1, num_features + 1):
            centroid = center_of_mass(binary, labeled, i)
            if centroid is None:
                continue
            yf, xf = centroid  # row, col (y, x)
            if np.isnan(xf) or np.isnan(yf):
                continue

            px = int(round(xf))
            py = int(round(yf))
            if not (0 <= px < W and 0 <= py < H):
                continue

            intensity = float(brightness[py, px])

            # Pixel -> NDC
            u = (px / W) * 2.0 - 1.0
            v = 1.0 - (py / H) * 2.0

            # Reconstruct camera-space direction (consistent with projection in _project_stars)
            dir_cam = normalize(np.array([u * f, v * f, 1.0], dtype=float))

            # Transform to world space
            dir_world = dir_cam[0] * right + dir_cam[1] * up + dir_cam[2] * forward
            dir_world = normalize(dir_world)

            # Simple distance estimate from intensity (optional); None if intensity==0
            if intensity > 0:
                z = depth_scale / math.sqrt(max(intensity, 1e-12))
                estimated_pos = np.asarray(cam_pos, dtype=float) + dir_world * z
            else:
                estimated_pos = None

            results.append((dir_world.astype(float), None if estimated_pos is None else estimated_pos.astype(float),
                            (px, py), intensity))

        return results


    """
        # Normalize and apply optional transformations
        if max_flux is None:
            max_flux = img.max()
            max_flux_16bit = grayscale_16bit.max()
            
        img = np.clip((img - min_flux) / (max_flux - min_flux), 0, 1)

        grayscale_16bit = np.clip((grayscale_16bit - min_flux) / (max_flux_16bit - min_flux), 0, 1)
    """
    @staticmethod
    def postprocess_image(image_data: np.ndarray,
                          grayscale_16bit: np.ndarray,
                          saturation_limit: float,
                          log_scale: bool, min_flux=0.0, max_flux=None) -> Tuple[np.ndarray, np.ndarray]:

        if max_flux is None:
            max_flux = saturation_limit

        if log_scale:
            #image_data = np.log1p(image_data) / np.log1p(max_flux)
            #rayscale_16bit = np.log1p(grayscale_16bit) / np.log1p(max_flux)

            image_data = np.log1p(np.clip(image_data, min_flux, max_flux)) / np.log1p(max_flux)
            grayscale_16bit = np.log1p(np.clip(grayscale_16bit, min_flux, max_flux)) / np.log1p(max_flux)
        else:
            image_data = np.clip(image_data, min_flux, max_flux)
            grayscale_16bit = np.clip(grayscale_16bit, min_flux, max_flux)

        return image_data, grayscale_16bit

    @staticmethod
    def apply_psf_and_blooming(psf, img: np.ndarray, grayscale_16bit: np.ndarray, blooming_factor: float,
                               saturation_limit: float) -> Tuple[np.ndarray, np.ndarray]:
        for c in range(3):
            img[:, :, c] = convolve(img[:, :, c], psf, mode="same")

        grayscale_16bit = convolve(grayscale_16bit, psf, mode="same")

        if blooming_factor > 0.0:
            overexposed = img > saturation_limit
            overexposed_grayscale_16bit = grayscale_16bit > saturation_limit

            for c in range(3):
                blooming = gaussian_filter(overexposed[:, :, c].astype(float), sigma=blooming_factor)
                img[:, :, c] += blooming * (saturation_limit / 2)

            blooming_gray = gaussian_filter(overexposed_grayscale_16bit.astype(float), sigma=blooming_factor)
            grayscale_16bit += blooming_gray * (saturation_limit / 2)

        return img, grayscale_16bit

    @property
    def area(self) -> float:
        return math.pi * (self.aperture_diameter / 2.0) ** 2

    def expected_signal_photons(self, star: Star, distance_m: float) -> float:
        bol_flux = star.flux_at(distance_m)
        frac = star.band_fraction(self.band_center_m, self.band_width_m)
        band_flux = bol_flux * frac
        photon_energy = h * c / self.band_center_m
        incident_energy = band_flux * self.area * self.throughput * self.integration_time
        incident_photons = incident_energy / photon_energy if photon_energy > 0 else 0.0
        detected = incident_photons * self.qe
        return float(detected)

    def snr(self, star: Star, distance_m: float, pointing_offset_deg: float = 0.0) -> float:
        # heavy penalty if outside FoV
        if pointing_offset_deg > self.fov_deg:
            return 0.0
        signal = self.expected_signal_photons(star, distance_m)
        background = self.background_photons_per_s * self.integration_time
        var = signal + background + (self.read_noise_e ** 2)
        return 0.0 if var <= 0 else signal / math.sqrt(var)

    def detection_probability(self, snr: float) -> float:
        # logistic curve centered near snr=5
        return 1.0 / (1.0 + math.exp(-0.8 * (snr - 5.0)))

    def _extract_arrays(self, sf_or_positions, temperatures=None, luminosities=None, radii=None):
        """
        Accept either:
          - a StarField-like object (common attribute names are checked), or
          - explicit arrays (positions, temperatures, luminosities, radii).
        Returns positions (N,3), temps, lums, rads as numpy arrays.
        """
        if temperatures is None and hasattr(sf_or_positions, "__class__") and isinstance(sf_or_positions, StarField):
            # StarField created in this project; inspect common attribute names
            sf = sf_or_positions
            if hasattr(sf, "positions"):
                positions = np.asarray(sf.positions)
            elif hasattr(sf, "star_positions"):
                positions = np.asarray(sf.star_positions)
            else:
                raise ValueError("StarField has no 'positions' or 'star_positions' attribute")

            temps = np.asarray(getattr(sf, "temperatures", getattr(sf, "star_temperatures", np.zeros(len(positions)))))
            lums = np.asarray(getattr(sf, "luminosities", getattr(sf, "star_luminosities", np.ones(len(positions)))))
            rads = np.asarray(getattr(sf, "radii", getattr(sf, "star_radii", np.zeros(len(positions)))))
        else:
            positions = np.asarray(sf_or_positions)
            temps = np.asarray(temperatures) if temperatures is not None else np.zeros(len(positions))
            lums = np.asarray(luminosities) if luminosities is not None else np.ones(len(positions))
            rads = np.asarray(radii) if radii is not None else np.zeros(len(positions))

        return positions, temps, lums, rads

    @staticmethod
    def show_interactive_starfield(star_field, fov_deg, max_points=25000, downsample_seed=0,
                                   show_axes=True, title="Star Field",
                                   show_camera=False, cam_pos=None, cam_dir=None, up_hint=None):
        """
        Display interactive 3D star field.
        - Accepts a StarField instance or arrays.
        - If show_camera True and cam_pos/cam_dir provided (or sensor passed), draws camera arrow and frustum.
        - Downsamples if more than max_points.
        """
        positions, temps, lums, rads = (
            star_field.star_positions,
            star_field.star_temperatures,
            star_field.star_luminosities,
            star_field.star_radii,
        )

        N = len(positions)
        if N == 0:
            raise ValueError("No star positions provided")

        if N > max_points:
            rng = np.random.default_rng(downsample_seed)
            idx = rng.choice(N, size=max_points, replace=False)
            positions = positions[idx]
            temps = temps[idx]
            lums = lums[idx]
            rads = rads[idx]

        # Colors by temperature (normalized)
        tmin, tmax = np.nanmin(temps), np.nanmax(temps)
        if tmin == tmax:
            color = np.ones_like(temps) * 0.5
        else:
            color = (temps - tmin) / max(1e-12, (tmax - tmin))

        # Sizes by log luminosity
        safe_lums = np.clip(lums, a_min=1e-12, a_max=None)
        logl = np.log10(safe_lums)
        size_norm = (logl - logl.min()) / max(1e-12, (logl.max() - logl.min()))
        sizes = 2 + size_norm * 10

        hover = [
            f"Idx: {i}<br>Dist (pc): {np.linalg.norm(p) / PC_TO_M:.3f}<br>Temp: {t:.0f} K<br>Lum: {l:.3e}<br>Rad: {r:.3e}"
            for i, (p, t, l, r) in enumerate(zip(positions, temps, lums, rads))
        ]

        scatter = go.Scatter3d(
            x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
            mode="markers",
            marker=dict(size=sizes, color=color, colorscale="Turbo", opacity=0.9, sizemode="diameter"),
            hoverinfo="text",
            hovertext=hover,
            name="Stars"
        )

        data = [scatter]

        # Camera visualization (arrow + simple frustum)
        if show_camera:
            fov_deg = fov_deg

            if cam_pos is None:
                cam_pos = np.array([0.0, 0.0, 0.0])
            if cam_dir is None:
                cam_dir = np.array([0.0, 0.0, 1.0])
            if up_hint is None:
                up_hint = np.array([0.0, 1.0, 0.0])

            cam_pos = np.asarray(cam_pos, dtype=float)
            forward = cam_dir / np.linalg.norm(cam_dir)
            right = np.cross(forward, up_hint)
            if np.linalg.norm(right) < 1e-12:
                # choose another up if degenerate
                up_hint2 = np.array([0.0, 1.0, 0.0]) if not np.allclose(up_hint, [0, 1, 0]) else np.array(
                    [1.0, 0.0, 0.0])
                right = np.cross(forward, up_hint2)
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)

            # Arrow showing camera direction
            arrow_len = np.linalg.norm(positions, axis=1).mean() * 0.05 if len(positions) > 0 else 1.0
            p1 = cam_pos
            p2 = cam_pos + forward * arrow_len
            camera_line = go.Scatter3d(x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
                                       mode="lines+markers", line=dict(color="red", width=4), marker=dict(size=2),
                                       name="Camera")
            data.append(camera_line)

            # Simple frustum: four rays at fov/2 in the forward plane
            half = np.tan(np.radians(fov_deg) / 2.0)
            depth = arrow_len * 4.0
            corners = [
                forward * depth + (right * sx + up * sy) * (half * depth)
                for sx, sy in [(-1, -1), (-1, 1), (1, 1), (1, -1)]
            ]
            frustum_x = []
            frustum_y = []
            frustum_z = []
            for c in corners:
                frustum_x += [cam_pos[0], cam_pos[0] + c[0], None]
                frustum_y += [cam_pos[1], cam_pos[1] + c[1], None]
                frustum_z += [cam_pos[2], cam_pos[2] + c[2], None]
            frustum = go.Scatter3d(x=frustum_x, y=frustum_y, z=frustum_z, mode="lines",
                                   line=dict(color="orange", width=2), name="Frustum")
            data.append(frustum)

        layout = go.Layout(
            title=title,
            scene=dict(
                xaxis=dict(visible=show_axes, title="X (m)"),
                yaxis=dict(visible=show_axes, title="Y (m)"),
                zaxis=dict(visible=show_axes, title="Z (m)"),
                aspectmode="data"
            ),
            margin=dict(l=0, r=0, b=0, t=30),
        )

        fig = go.Figure(data=data, layout=layout)
        fig.show()