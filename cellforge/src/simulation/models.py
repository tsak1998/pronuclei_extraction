from dataclasses import dataclass
from warp import quatf, vec3
"""
E: Young's module
v: Poisson's ratio
k_mu: E/(2(1+v))
k_lambda:  Ev/((1+v)(1-2v))
"""


@dataclass
class MeshPhysicalProperties:

    density: float
    v: float
    E: float
    k_damp: float  # damping stiffness
    k_mu: float | None = None  # first elastic Lame parameter
    k_lambda: float | None = None  # second elastic Lame parameter (shear Modulus)

    def __post_init__(self):
        # Calculate Lame parameters if not given.
        if self.k_mu is None:
            self.k_mu = self.E / (2 * (1 + self.v))

        if self.k_lambda is None:
            self.k_lambda = (self.E * self.v) / ((1 + self.v) *
                                                 (1 - 2 * self.v))


@dataclass
class MeshPostionProperties:
    pos: vec3
    rot: quatf
    vel: vec3
    scale: float = 1.0


@dataclass
class TetrahedralMesh:
    vertices: list[float]
    tetra_indices: list[int]


@dataclass
class WarpSoftMesh:
    physical_properties: MeshPhysicalProperties
    position: MeshPostionProperties
    mesh: TetrahedralMesh
