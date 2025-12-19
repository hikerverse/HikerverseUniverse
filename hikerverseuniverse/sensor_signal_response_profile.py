
from dataclasses import dataclass

@dataclass(frozen=True)
class SensorSignalResponseProfile:
    optical_response: float
    radar_response: float
    gravimetric_response: float
    magnetometric_response: float
    subspace_resonance_response: float

    def get_optical_response(self) -> float:
        return self.optical_response

    def get_radar_response(self) -> float:
        return self.radar_response

    def get_gravimetric_response(self) -> float:
        return self.gravimetric_response

    def get_magnetometric_response(self) -> float:
        return self.magnetometric_response

    def get_subspace_resonance_response(self) -> float:
        return self.subspace_resonance_response
