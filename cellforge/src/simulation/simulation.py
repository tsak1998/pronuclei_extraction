import datetime
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


class Simulation:

    def __init__(
        self,
        E: float,
        v: float,
        damping: float,
        density: float
    ):
        self.mesh_generator = MeshGenerator()
        self.sim_width = 8
        self.sim_height = 8

        self.frame_dt = 1.0 / config.fps
        self.sim_substeps = 64  # Increased substeps for better accuracy
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.sim_iterations = 1
        self.sim_relaxation = 1.0
        self.profiler = {}

        builder = wp.sim.ModelBuilder()

        # Set a reasonable default particle
        
        # builder.default_particle_radius = 0.1
        sphere_resolution: int = 15
        radius = 1
        sphere_mesh = self.mesh_generator.generate_sphere_mesh(
            radius=radius,
            center=(0,10,10),
            theta_resolution=sphere_resolution,
            phi_resolution=sphere_resolution)

       
     

        mesh_physical_props = MeshPhysicalProperties(density=density,
                                                     E=E,
                                                     v=v,
                                                     k_damp=damping)
        
        for i in range(1):
            mesh_position = MeshPostionProperties(pos=wp.vec3(
                0, 10, 10),
                                                  rot=wp.quat_identity(),
                                                  scale=1.0,
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

        self.model = builder.finalize()
        self.model.enable_tri_collisions = True

        self.model.ground = True


        self.integrator = wp.sim.SemiImplicitIntegrator()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        stage_path = f'{config.experiment_path}/{E}_{v}_{density}_{damping}.usd'
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
    
    parser.add_argument("--num_frames",
                        type=int,
                        default=config.simul_frames,
                        help="Total number of frames.")

    args = parser.parse_known_args()[0]

    Es = [90e5]
    vs = [ 0.2]
    densities = [400]
    damping = [0.0]

    for E in Es:
        for v in vs:
            for density in densities:
                for d in damping:
                    try:
                        with wp.ScopedDevice(args.device):
                            example = Simulation(E,v,d,density)

                            for fr in range(args.num_frames):
                                example.step()
                                example.render()

                            if example.renderer:
                                example.renderer.save()
                    except Exception as e:
                        print(e)
                        print(E,v,d,density)
                        continue