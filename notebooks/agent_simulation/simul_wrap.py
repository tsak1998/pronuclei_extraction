import warp as wp
import warp.sim
import warp.sim.render

import pyvista as pv

# Create a sphere surface mesh
sphere_surface = pv.Sphere(radius=1.0, theta_resolution=5, phi_resolution=5)

# Convert to a tetrahedral mesh
tetra_mesh = sphere_surface.delaunay_3d()
import numpy as np


class Example:

    def __init__(self, stage_path="example_rigid_soft_contact.usd"):
        self.sim_width = 8
        self.sim_height = 8

        fps = 60
        self.frame_dt = 1.0 / fps
        self.sim_substeps = 8  # Increased substeps for better accuracy
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.sim_iterations = 1
        self.sim_relaxation = 1.0
        self.profiler = {}

        builder = wp.sim.ModelBuilder()

        # Set a reasonable default particle radius
        builder.default_particle_radius = 0.5  # Adjust as needed

        # Add the first soft grid positioned at (0, 0, 0)
        # builder.add_soft_grid(
        #     pos=wp.vec3(0.0, 0.0, 0.0),
        #     rot=wp.quat_identity(),
        #     vel=wp.vec3(0.0, 0.0, 0.0),
        #     dim_x=10,  # Adjust dimensions as needed
        #     dim_y=5,
        #     dim_z=10,
        #     cell_x=0.1,
        #     cell_y=0.1,
        #     cell_z=0.1,
        #     density=1000.0,  # Adjusted density
        #     k_mu=50000.0,
        #     k_lambda=20000.0,
        #     k_damp=0.2,
        # )

        # Add a soft sphere
        sphere_surface = pv.Sphere(radius=1,
                                   theta_resolution=15,
                                   phi_resolution=15)

        # tetra_mesh = sphere_surface.delaunay_3d()

        # Extract vertices and tetrahedral cells
        # vertices = tetra_mesh.points
        # tetra_indices = tetra_mesh.cells_dict[pv.CellType.TETRA].flatten()
        import tetgen
        # sphere = pv.Sphere()
        tet = tetgen.TetGen(sphere_surface)
        vertices, elems = tet.tetrahedralize()
        tetra_indices = elems.flatten()
        # Check if tetrahedral cells are present
        if len(tetra_indices) == 0:
            raise ValueError("No tetrahedral cells found in the mesh.")

        # Add the sphere to the simulation
        builder.add_soft_mesh(
            pos=wp.vec3(0.0, 25.0, 0.0),  # Position the sphere in space
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, -5.0, 0.0),  # Initial downward velocity
            vertices=vertices,
            indices=tetra_indices,
            density=1000.0,  # Adjusted density
            k_mu=50000.0,
            k_lambda=20000.0,
            k_damp=0.2,
        )

        builder.soft_contact_ke = 1.0e5
        builder.soft_contact_kd = 1.0e2
        builder.soft_contact_kf = 1.0e4
        builder.soft_contact_mu = 0.5

        # Optionally, enable particle-particle collisions
        builder.particle_contact = False
        builder.particle_contact_ke = 1.0e5
        builder.particle_contact_kd = 1.0e2
        builder.particle_contact_kf = 1.0e4
        builder.particle_contact_mu = 0.5

        self.model = builder.finalize()

        # Apply gravity to the model
        self.model.gravity = wp.vec3(0.0, -9.81, 0.0)

        # Enable ground plane if needed
        self.model.ground = True
        self.model.enable_tri_collisions = True

        self.integrator = wp.sim.SemiImplicitIntegrator()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        if stage_path:
            self.renderer = wp.sim.render.SimRenderer(self.model,
                                                      stage_path,
                                                      scaling=1.0)
        else:
            self.renderer = None

        self.use_cuda_graph = wp.get_device().is_cuda
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _s in range(self.sim_substeps):
            # Collision detection
            wp.sim.collide(self.model, self.state_0)

            # Clear forces before applying new ones
            self.state_0.clear_forces()
            self.state_1.clear_forces()

            # Integrate
            self.integrator.simulate(self.model, self.state_0, self.state_1,
                                     self.sim_dt)

            # Swap states
            (self.state_0, self.state_1) = (self.state_1, self.state_0)

    def step(self):
        with wp.ScopedTimer("step", dict=self.profiler):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
            self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device",
                        type=str,
                        default=None,
                        help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="example_rigid_soft_contact.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_frames",
                        type=int,
                        default=200,
                        help="Total number of frames.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path)

        for _ in range(args.num_frames):
            example.step()
            example.render()

        if example.renderer:
            example.renderer.save()
