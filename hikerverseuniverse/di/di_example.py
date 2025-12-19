# python
import numpy as np

from hikerverseuniverse.celestials.star_field import StarField
from hikerverseuniverse.di.di_lib import Container
from hikerverseuniverse.di.di_lib import Inject
from hikerverseuniverse.di.di_lib import inject_constructor
from hikerverseuniverse.di.di_registry import IService, ServiceImpl, IOpticalSensor
from hikerverseuniverse.sensor_physics.optical_sensor_implementation import OpticalSensorImpl
from hikerverseuniverse.utils.math_utils import gaussian_psf

# create container and register
c = Container()
c.register(IService, ServiceImpl, singleton=True)
c.register(IOpticalSensor, OpticalSensorImpl, singleton=True)


# constructor injection
@inject_constructor(c)
class ConsumerA:
    def __init__(self, svc: IService, src2: IOpticalSensor):
        self.svc = svc
        self.svc2 = src2


a = ConsumerA()  # svc auto-resolved
print(a.svc.do())


# attribute injection
class ConsumerB:
    svc: IService = Inject()
    svc2: IOpticalSensor = Inject()


b = ConsumerB()
c.inject_into(b)  # fills b.svc
print(b.svc.do())

b.svc2.take_image(psf=gaussian_psf(3, 1),
                  star_field=StarField([[1, 1, 1]],[[1, 1, 1]],[[1, 1, 1]], [[1, 1, 1]]),
                  fov_deg=45, resolution=(512, 512),
                  aperture_diameter=1, band_center_m=550e-9,
                  telescope_position=np.array([0, 0, 0]), camera_direction=np.array([0, 0, -1]),
                  up_hint=np.array([0, 1, 0]),
                  threshold=5e-18,
                  exposure=1, saturation_limit=1e-10, blooming_factor=0.0,
                  log_scale=False, min_flux=0.0, max_flux=None, gain=1)
