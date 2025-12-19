from typing import Tuple

import numpy as np

from hikerverseuniverse.utils.math_utils import normalize


def compute_camera_basis(camera_direction: np.ndarray, up_hint: np.ndarray) -> Tuple[
                                                        np.ndarray, np.ndarray, np.ndarray]:
    forward = normalize(camera_direction)
    right = normalize(np.cross(forward, up_hint))
    up = np.cross(right, forward)
    return forward, right, up
