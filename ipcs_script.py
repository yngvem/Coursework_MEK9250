import argparse
from math import ceil

import numpy as np
import fenics as pde
import matplotlib.pyplot as plt
import mshr
from tqdm import trange
from ipcs import ImplicitSolver, ExplicitSolver, simulate_stokes

# FEniCS setup
pde.parameters['linear_algebra_backend'] = "PETSc"
pde.set_log_level(40) # warning

# Geometry:
width: "m" = 2.2
diameter: "m" = 0.1
radius: "m" = diameter / 2
height: "m" = 0.15 + 0.16 + diameter
cylinder_center_x: "m" = 0.15 + radius
cylinder_center_y: "m" = 0.15 + radius
cylinder_radius: "m" = 0.05
cylinder_center: ("m", "m") = (0.2 ,0.2)

# Physical parameters
max_inflow_velocity: "m/s" = 1.5
viscosity: "m^2/s" = 1e-3
density: "kg/m^3" = 1

# Define geometry and specify boundaries
flow_channel = (
      mshr.Rectangle(pde.Point(0, 0), pde.Point(width, height)) 
    - mshr.Circle(pde.Point(cylinder_center_x, cylinder_center_y), radius)
)
at_inflow  = 'near(x[0], 0)'
at_outflow = f'near(x[0], {width})'
at_cylinder = (
    f'(x[0] >= ({(cylinder_center_x - radius) / 2})) &&'
    f'(x[0] <= ({(cylinder_center_x + radius + width) / 2})) &&'
    f'(x[1] >= ({(cylinder_center_y - radius) / 2})) &&'
    f'(x[1] <= ({(cylinder_center_x + radius + height) / 2})) &&'
    'on_boundary'
)
noslip   = f'near(x[1], 0) || near(x[1], {height}) || ({at_cylinder})'

# Set inflow profile
inflow_profile = pde.Expression(
    ("4*Um * x[1] * (H - x[1])/(H*H)", 0),
    Um=max_inflow_velocity,
    H=height,
    degree=2,
)

# Define boundaries
def create_boundary_conditions(height, max_inflow_velocity, velocity_space, pressure_space,):
    V, Q = velocity_space, pressure_space
    bcu_noslip  = pde.DirichletBC(V, pde.Constant((0, 0)), noslip)
    bcu_inflow  = pde.DirichletBC(V, inflow_profile, at_inflow)
    bcp_outflow = pde.DirichletBC(Q, pde.Constant(0), at_outflow)
    bcu = [bcu_noslip, bcu_inflow]
    bcp = [bcp_outflow]
    return bcu, bcp


def create_function_spaces(mesh, velocity_order=2, pressure_order=1):
    P2 = pde.VectorElement('P', mesh.ufl_cell(), velocity_order)
    P1 = pde.FiniteElement('P', mesh.ufl_cell(), pressure_order)

    V = pde.FunctionSpace(mesh, P2)
    Q = pde.FunctionSpace(mesh, P1)
    TH = pde.FunctionSpace(mesh, P2*P1)
    return V, Q, TH


def compute_timestep(scale, velocity, resolution, method):
    if method.lower() == 'implicit':
        return scale / (velocity * resolution)
    elif method.lower() == 'explicit':
        return scale / (velocity * resolution**2)
    else:
        raise ValueError("method must be either explicit or implicit")


if __name__ == "__main__":
    # Simulation parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestep_scale", default=1, type=float, help="Coefficient used to compute dt")
    parser.add_argument("--method", default="implicit", type=str, help="implicit or explicit")
    parser.add_argument("--initial_resolution", default=64, type=int, help="Resolution used to obtain steady state")
    parser.add_argument("--resolution", default=128, type=int, help="Resolution used to compute drag coefficient and pressure difference")
    parser.add_argument("--initial_simulation_time", default=1.5, type=float, help="Time to simulate on low resolution mesh to obtain steady state")
    parser.add_argument("--simulation_time", default=0.5, type=float, help="Simulation time used to compute drag coefficient and pressure difference")
    parser.add_argument("--plot_frequency", default=10, type=float, help="Plot update frequency, if negative, then there will not be any plotting")
    parser.add_argument("--velocity_order", default=2, type=int, help="Order of interpolating polynomial used for the velocity component")

    args = parser.parse_args()

    timestep_scale = args.timestep_scale
    method = args.method

    initial_resolution = args.initial_resolution
    final_resolution = args.resolution

    initial_simulation_time = args.initial_simulation_time
    simulation_time = args.simulation_time

    velocity_order = args.velocity_order
    plot_frequency = args.plot_frequency

    SolverClass = {'explicit': ExplicitSolver, 'implicit': ImplicitSolver}[method.lower()]

    # Mesh and time-steps
    low_res_mesh = mshr.generate_mesh(flow_channel, initial_resolution)
    dt_lr = compute_timestep(timestep_scale, max_inflow_velocity, initial_resolution, method)
    num_steps_lr = ceil(initial_simulation_time / dt_lr)

    mesh = mshr.generate_mesh(flow_channel, final_resolution)
    dt = compute_timestep(timestep_scale, max_inflow_velocity, final_resolution, method)
    num_steps = ceil(simulation_time / dt)

    high_pressure_point = (0.15, 0.2)
    low_pressure_point = (0.25, 0.2)
    pressure_points = (high_pressure_point, low_pressure_point)

    # Function space and BCs
    V_lr, Q_lr, TH_lr = create_function_spaces(low_res_mesh, velocity_order=velocity_order)
    V, Q, _ = create_function_spaces(mesh, velocity_order=velocity_order)
    bcu_lr, bcp_lr = create_boundary_conditions(height, max_inflow_velocity, V_lr, Q_lr)
    bc_stokes = create_boundary_conditions(height, max_inflow_velocity, *TH_lr.split())[0]
    bcu, bcp = create_boundary_conditions(height, max_inflow_velocity, V, Q)

    # Initialise with Stokes
    u_stokes_, p_stokes_ = simulate_stokes(TH_lr, bc_stokes)

    # Solve with low resolution to obtain steady state
    print("Low resolution solver to obtain steady state...", flush=True)
    low_res_solver = SolverClass(low_res_mesh, V_lr, Q_lr, dt_lr, viscosity, density, plot_frequency)
    low_res_solver.simulate(num_steps_lr, u_stokes_, p_stokes_, bcu_lr, bcp_lr, at_cylinder, pressure_points)

    # Solve with high resolution to obtain better estimates
    print("High resolution solver to accurately estimate drag and pressure difference...", flush=True)
    solver = SolverClass(mesh, V, Q, dt, viscosity, density, plot_frequency)
    solver.simulate(num_steps, low_res_solver.u_, low_res_solver.p_, bcu, bcp, at_cylinder, pressure_points)

    U_bar = 2*max_inflow_velocity/3
    max_drag_coef = -2*min(solver.drag_forces) / density / diameter / (U_bar**2)
    print("Maximum drag coefficient:", max_drag_coef)
    print("Maximum pressure difference:", max(solver.pressure_differences))
