# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Rigid FEM
#
# Shows how to set up a rigid sphere colliding with an FEM beam
# using wp.sim.ModelBuilder().
#
###########################################################################

import warp as wp
import warp.sim
import warp.sim.render

import pyvista as pv
import tetgen

import yaml

from cellforge.src.config import Config
from cellforge.src.simulation.mesh_generator import MeshGenerator
from cellforge.src.simulation.models import MeshPhysicalProperties, MeshPostionProperties

with open('src/conf/simulation.yaml') as f:

    try:
        config = Config(**yaml.safe_load(f))
    except yaml.YAMLError as exc:
        print(exc)


@wp.kernel
def apply_buoyancy_and_drag(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=float),
    forces: wp.array(dtype=wp.vec3),
    fluid_density: float,
    gravity: wp.vec3,
    drag_coefficient: float,
    volume_per_particle: float,
):
    tid = wp.tid()

    # Get particle properties
    mass = masses[tid]
    vel = velocities[tid]

    # Buoyancy force per particle
    # Buoyant force acts upward, opposite to gravity
    buoyancy_force = -fluid_density * volume_per_particle * gravity

    # Drag force per particle
    relative_velocity = vel  # Assuming fluid at rest
    speed = wp.length(relative_velocity)
    if speed > 0.0:
        drag_force = -0.5 * fluid_density * drag_coefficient * volume_per_particle * speed * (
            relative_velocity / speed)
    else:
        drag_force = wp.vec3()

    # Total force
    total_force = buoyancy_force + drag_force

    # Apply forces
    forces[tid] += total_force


class Example:

    def __init__(self, stage_path="example_rigid_soft_contact.usd"):
        self.sim_width = 8
        self.sim_height = 8

        fps = 60
        self.frame_dt = 1.0 / fps
        self.sim_substeps = config.sim_substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.sim_iterations = 1
        self.sim_relaxation = 1.0
        self.profiler = {}

        builder = wp.sim.ModelBuilder()
        builder.default_particle_radius = 0.01

<<<<<<< HEAD

        builder.add_soft_grid(
            pos=wp.vec3(0.0, 5.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=20,
            dim_y=10,
            dim_z=10,
            cell_x=0.1,
            cell_y=0.1,
            cell_z=0.1,
            density=100.0,
            k_mu=50000.0,
            k_lambda=20000.0,
            k_damp=0.0,
        )


        builder.add_soft_grid(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=20,
            dim_y=10,
            dim_z=10,
            cell_x=0.1,
            cell_y=0.1,
            cell_z=0.1,
            density=100.0,
            k_mu=50000.0,
            k_lambda=20000.0,
            k_damp=0.0,
        )
=======
        soft_body_dim = 30  # Number of cells in each dimension
        soft_body_cell_size = 0.1  # Size of each cell
        soft_body_position = (0.0, 10.0, 0.0)
        soft_body_density = 500.0  # Density of the soft body (kg/m^3), adjust this
>>>>>>> edc2261 (small scale simulation)

        # builder.add_soft_grid(
        #     pos=wp.vec3(*soft_body_position),
        #     rot=wp.quat_identity(),
        #     vel=wp.vec3(0.0, 0.0, 0.0),
        #     dim_x=soft_body_dim,
        #     dim_y=soft_body_dim,
        #     dim_z=soft_body_dim,
        #     cell_x=soft_body_cell_size,
        #     cell_y=soft_body_cell_size,
        #     cell_z=soft_body_cell_size,
        #     density=soft_body_density,
        #     k_mu=5e4,
        #     k_lambda=2e4,
        #     k_damp=0.0,
        # )
        self.mesh_generator = MeshGenerator()

        builder.default_particle_radius = 1e-8
        sphere_resolution: int = 16
        radius = 1e-4
        sphere_mesh = self.mesh_generator.generate_sphere_mesh(
            radius=radius,
            theta_resolution=sphere_resolution,
            phi_resolution=sphere_resolution)

        E = 2e3
        v = 0.49
        damping = 0.0
        density = 1000

        mesh_physical_props = MeshPhysicalProperties(density=density,
                                                     E=E,
                                                     v=v,
                                                     k_damp=damping)

        self.object_volume = (soft_body_dim *
                              soft_body_cell_size)**3  # Volume of the cube
        mesh_position = MeshPostionProperties(pos=wp.vec3(0.0, 1e-1, 0.0),
                                              rot=wp.quat_identity(),
                                              scale=5.0,
                                              vel=wp.vec3(0.0, 0.0, 0.0))
        builder.add_soft_mesh(
            pos=mesh_position.pos,
            rot=mesh_position.rot,
            scale=mesh_position.scale,
            vel=mesh_position.pos,
            vertices=sphere_mesh.vertices,
            indices=sphere_mesh.tetra_indices,
            density=mesh_physical_props.density,
            k_mu=mesh_physical_props.k_mu,
            k_lambda=mesh_physical_props.k_lambda,
            k_damp=mesh_physical_props.k_damp,
        )
        mesh_position = MeshPostionProperties(pos=wp.vec3(1e-3, 2e-1, 0.0),
                                              rot=wp.quat_identity(),
                                              scale=1.0,
                                              vel=wp.vec3(0.0, 0.0, 0.0))
        # builder.add_soft_mesh(
        #     pos=mesh_position.pos,
        #     rot=mesh_position.rot,
        #     scale=mesh_position.scale,
        #     vel=mesh_position.pos,
        #     vertices=sphere_mesh.vertices,
        #     indices=sphere_mesh.tetra_indices,
        #     density=mesh_physical_props.density,
        #     k_mu=mesh_physical_props.k_mu,
        #     k_lambda=mesh_physical_props.k_lambda,
        #     k_damp=mesh_physical_props.k_damp,
        # )

        self.model = builder.finalize()
        self.model.enable_tri_collisions = True
        self.model.ground = True
        self.model.enable_tri_collisions = True
        self.model.soft_contact_ke = 1.0e3
        self.model.soft_contact_kd = 0.0
        self.model.soft_contact_kf = 1.0e3

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

        # Fluid properties for custom forces
        self.fluid_density = 1000.0  # kg/m^3
        self.drag_coefficient = 0.5  # Adjust as needed
        self.gravity = wp.vec3(0.0, -9.81, 0.0)

    def simulate(self):
        for _s in range(self.sim_substeps):
            wp.sim.collide(self.model, self.state_0)

            self.state_0.clear_forces()
            self.state_1.clear_forces()

            num_particles = len(self.model.particle_q)

            # Compute volume per particle (approximate)

            total_mass = num_particles * 0.1  #sum(self.model.particle_mass)
            total_volume = total_mass / self.fluid_density
            volume_per_particle = self.object_volume / num_particles

            # wp.launch(kernel=apply_buoyancy_and_drag,
            #           dim=num_particles,
            #           inputs=[
            #               self.state_0.particle_q,
            #               self.state_0.particle_qd,
            #               self.model.particle_mass,
            #               self.state_0.particle_f,
            #               self.fluid_density,
            #               self.gravity,
            #               self.drag_coefficient,
            #               volume_per_particle,
            #           ],
            #           device=self.model.device)
            self.integrator.simulate(self.model, self.state_0, self.state_1,
                                     self.sim_dt)

            # swap states
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
                        default=300,
                        help="Total number of frames.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path)

        for _ in range(args.num_frames):
            example.step()
            example.render()

        if example.renderer:
            example.renderer.save()
