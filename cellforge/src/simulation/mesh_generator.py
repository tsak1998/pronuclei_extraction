import pyvista as pv
import tetgen

from cellforge.src.simulation.models import (MeshPhysicalProperties,
                                             MeshPostionProperties,
                                             TetrahedralMesh, WarpSoftMesh)


class MeshGenerator:

    def generate_sphere_mesh(self, radius: int,center: tuple[float], theta_resolution: int,
                             phi_resolution: int) -> TetrahedralMesh:

        sphere_surface = pv.Sphere(radius=radius,
                                   center=center,
                                   theta_resolution=theta_resolution,
                                   phi_resolution=phi_resolution)

        tet = tetgen.TetGen(sphere_surface)
<<<<<<< HEAD
        vertices, elems = tet.tetrahedralize( mindihedral=20, minratio=1.5)
=======
        vertices, elems = tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5)
>>>>>>> edc2261 (small scale simulation)
        tetra_indices = elems.flatten()

        return TetrahedralMesh(vertices=vertices, tetra_indices=tetra_indices)

    def generate_soft_mesh(
            self, mesh: TetrahedralMesh, position: MeshPostionProperties,
            physical_properties: MeshPhysicalProperties) -> WarpSoftMesh:

        return WarpSoftMesh(mesh=mesh,
                            position=position,
                            physical_properties=physical_properties)
