import pyvista as pv
import tetgen

from cellforge.src.simulation.models import (MeshPhysicalProperties,
                                             MeshPostionProperties,
                                             TetrahedralMesh, WarpSoftMesh)


class MeshGenerator:

    def generate_sphere_mesh(self, radius: int, theta_resolution: int,
                             phi_resolution: int) -> TetrahedralMesh:

        sphere_surface = pv.Sphere(radius=radius,
                                   theta_resolution=theta_resolution,
                                   phi_resolution=phi_resolution)

        tet = tetgen.TetGen(sphere_surface)
        vertices, elems = tet.tetrahedralize()
        tetra_indices = elems.flatten()

        return TetrahedralMesh(vertices=vertices, tetra_indices=tetra_indices)

    def generate_soft_mesh(
            self, mesh: TetrahedralMesh, position: MeshPostionProperties,
            physical_properties: MeshPhysicalProperties) -> WarpSoftMesh:

        return WarpSoftMesh(mesh=mesh,
                            position=position,
                            physical_properties=physical_properties)
