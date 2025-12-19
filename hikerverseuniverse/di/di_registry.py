
from typing import Protocol, Tuple

import numpy as np

from hikerverseuniverse.di.di_lib import Container
from hikerverseuniverse.sensor_physics.optical_sensor_implementation import OpticalSensorImpl


class IService(Protocol):
    def do(self) -> str: ...


class ServiceImpl:
    def do(self) -> str:
        return "ok"


class IOpticalSensor(Protocol):
    def get_star_field(self) -> bytes: ...

    def take_image(self, psf, star_field, fov_deg, resolution, aperture_diameter, band_center_m,
                   telescope_position: np.ndarray, camera_direction: np.ndarray, up_hint: np.ndarray,
                   threshold: float,
                   exposure: float, saturation_limit: float, blooming_factor: float = 0.0,
                   log_scale: bool = False, min_flux=0.0, max_flux=None, gain=1) -> Tuple[np.ndarray, np.ndarray]: ...


# create container and register
c = Container()
c.register(IService, ServiceImpl, singleton=True)
c.register(IOpticalSensor, OpticalSensorImpl, singleton=True)
