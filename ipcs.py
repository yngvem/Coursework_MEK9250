from math import ceil

import numpy as np
import fenics as pde
import matplotlib.pyplot as plt
import mshr
from tqdm import trange


# Aliases for FEniCS names we will use
dx = pde.dx
ds = pde.ds
sym = pde.sym
grad = pde.grad
nabla_grad = pde.nabla_grad
div = pde.div

dot = pde.dot
inner = pde.inner

def epsilon(u): return sym(grad(u))
I = pde.Identity(2)


class IPCS:
    def __init__(self, mesh, velocity_space, pressure_space, dt, viscosity, density, plot_freq=-1):
        self.mesh = mesh
        self.velocity_space = velocity_space
        self.pressure_space = pressure_space
        self.dt = dt
        self.viscosity = viscosity
        self.density = density
        self.plot_freq = plot_freq

    def get_trial_functions(self):
        V, Q = self.velocity_space, self.pressure_space
        return pde.TrialFunction(V), pde.TrialFunction(Q)

    def get_test_functions(self):
        V, Q = self.velocity_space, self.pressure_space
        return pde.TestFunction(V), pde.TestFunction(Q)

    def _setup_traction(self, boundary):
        mu = pde.Constant(self.viscosity)
        n   = pde.FacetNormal(self.mesh)

        mf = pde.MeshFunction('size_t', self.mesh, 1)
        pde.CompiledSubDomain(boundary).mark(mf, 1)
        ds = pde.ds(subdomain_data=mf)
        sigma = (
                - self.p_*I
                + mu*epsilon(self.u_)
        )
        self.traction = dot(sigma, n)
        self.ds_traction = ds(1)

    def _setup_pressure_differences(self, high_pressure_boundary, low_pressure_boundary):
        mf = pde.MeshFunction('size_t', self.mesh, 1)
        pde.CompiledSubDomain(high_pressure_boundary).mark(mf, 1)
        pde.CompiledSubDomain(low_pressure_boundary).mark(mf, 2)
        ds = pde.ds(subdomain_data=mf)
        self.ds_high_pressure = ds(1)
        self.ds_low_pressure = ds(2)

    def _initialise_solution(self, initial_velocity, initial_pressure):
        # Setup initial conditions
        self.u_star_.vector()[:] = 0
        self.phi_.vector()[:] = 0
        self.u_.assign(pde.project(initial_velocity, self.velocity_space))
        self.p_.assign(pde.project(initial_pressure, self.pressure_space))

    def simulate(self, num_steps, initial_velocity, initial_pressure, bc_velocity, bc_pressure, traction_boundary, pressure_points):
        self._setup_variational_problems()
        self._setup_traction(traction_boundary)
        self._initialise_solution(initial_velocity, initial_pressure)
        self._setup_solvers(bc_velocity, bc_pressure)
        self.high_pressure_point, self.low_pressure_point = pressure_points

        self.t = 0
        self.time_points = []
        self.drag_forces = []
        self.lift_forces = []
        self.pressure_differences = []

        self.continue_simulation(num_steps)

    def continue_simulation(self, num_steps):
        fig, (velocity_ax, pressure_ax) = plt.subplots(2)
        for it in trange(num_steps):
            # Update current time
            self.t += self.dt
            self.update_solution()

            # Compute drag and lift
            self.time_points.append(self.t)

            # Forces
            ds = self.ds_traction
            self.drag_forces.append(pde.assemble(self.traction[0]*ds))
            self.lift_forces.append(pde.assemble(self.traction[1]*ds))

            # Pressure differences
            high_pressure = self.p_(*self.high_pressure_point)
            low_pressure = self.p_(*self.low_pressure_point)
            self.pressure_differences.append(high_pressure - low_pressure)

            # Plot solution
            if self.plot_freq > 0 and it % self.plot_freq == 0:
                velocity_ax.clear()
                pressure_ax.clear()
                plt.sca(velocity_ax)
                pde.plot(self.u_)
                plt.sca(pressure_ax)
                pde.plot(self.p_)
                plt.pause(0.1)


class ImplicitSolver(IPCS):
    def _setup_variational_problems(self):
        V, Q = self.velocity_space, self.pressure_space
        # Get trial and test functions
        u, p = self.get_trial_functions()
        u_star, phi = self.get_trial_functions()
        v, q = self.get_test_functions()

        # Define expressions used in variational forms
        n   = pde.FacetNormal(self.mesh)
        f   = pde.Constant((0, 0))
        k   = pde.Constant(self.dt)
        mu  = pde.Constant(self.viscosity)
        rho = pde.Constant(self.density)
        nu  = mu / rho

        # Define functions for solutions at previous and current time steps
        self.u_  = pde.Function(V)
        self.u_star_  = pde.Function(V)
        self.phi_ = pde.Function(Q)
        self.p_  = pde.Function(Q)
        u_, u_star_, phi_, p_ = self.u_, self.u_star_, self.phi_, self.p_

        # Define variational problem for step 1
        self.a1 = (
            dot(u_star, v) * dx
            + k * dot( dot(u_, nabla_grad(u_star)), v ) * dx
            + k * mu * inner(epsilon(u_star), epsilon(v)) * dx
        )
        self.L1 = (
            dot(u_, v) * dx
            - (k/rho) * dot(grad(p_), v) * dx
            - k * dot(f, v) * dx
        )

        # Define variational problem for step 2
        self.a2 = dot(grad(p), grad(q))*dx
        self.L2 = -(rho/k) * div(u_star_) * q * dx

        # Define variational problem for step 3
        self.a3 = dot(u, v)*dx
        self.L3 = dot(u_star_, v)*dx - (k/rho)*dot(grad(phi_), v)*dx

        # Define variational problem for step 4
        self.a4 = dot(p, q)*dx
        self.L4 = dot(p_ + phi_, q)*dx

    def _setup_solvers(self, bc_velocity, bc_pressure):
        A2 = pde.assemble(self.a2)
        A3 = pde.assemble(self.a3)
        A4 = pde.assemble(self.a4)
        for bc in bc_pressure:
            bc.apply(A2)
        solver2 = pde.LUSolver(A2)
        solver3 = pde.LUSolver(A3)
        solver4 = pde.LUSolver(A4)
        def update_solution():
            # Step 1: Tentative velocity step
            pde.solve(self.a1 == self.L1, self.u_star_, bcs=bc_velocity)

            # Step 2: Pressure correction step
            b2 = pde.assemble(self.L2)
            for bc in bc_pressure:
                bc.apply(b2)
            solver2.solve(self.phi_.vector(), b2)

            # Step 3: Velocity correction step
            b3 = pde.assemble(self.L3)
            solver3.solve(self.u_.vector(), b3)
            
            # Step 4: Pressure update
            b4 = pde.assemble(self.L4)
            solver4.solve(self.p_.vector(), b4)
        self.update_solution = update_solution


class ExplicitSolver(IPCS):
    def _setup_variational_problems(self):
        V, Q = self.velocity_space, self.pressure_space
        # Get trial and test functions
        u, p = self.get_trial_functions()
        u_star, phi = self.get_trial_functions()
        v, q = self.get_test_functions()

        # Define expressions used in variational forms
        n   = pde.FacetNormal(self.mesh)
        f   = pde.Constant((0, 0))
        k   = pde.Constant(self.dt)
        mu  = pde.Constant(self.viscosity)
        rho = pde.Constant(self.density)
        nu  = mu / rho

        # Define functions for solutions at previous and current time steps
        self.u_  = pde.Function(V)
        self.u_star_  = pde.Function(V)
        self.phi_ = pde.Function(Q)
        self.p_  = pde.Function(Q)
        u_, u_star_, phi_, p_ = self.u_, self.u_star_, self.phi_, self.p_

        # Define variational problem for step 1
        self.a1 = (
              dot(u, v) * dx
        )
        self.L1 = (
              dot(u_, v) * dx
            - k * dot( dot(u_, nabla_grad(u_)), v) * dx
            - (k/rho) * dot(grad(p_), v) * dx
            - k * mu * inner(sym(grad(u_)), sym(grad(v))) * dx
            + k * dot(f, v) * dx
        )

        # Define variational problem for step 2
        self.a2 = dot(grad(p), grad(q))*dx
        self.L2 = -(rho/k) * div(u_star_) * q * dx

        # Define variational problem for step 3
        self.a3 = dot(u, v)*dx
        self.L3 = dot(u_star_, v)*dx - (k/rho)*dot(grad(phi_), v)*dx

        # Define variational problem for step 4
        self.a4 = dot(p, q)*dx
        self.L4 = dot(p_ + phi_, q)*dx

    def _setup_solvers(self, bc_velocity, bc_pressure):
        A1 = pde.assemble(self.a1)
        A2 = pde.assemble(self.a2)
        A3 = pde.assemble(self.a3)
        A4 = pde.assemble(self.a4)

        for bc in bc_velocity:
            bc.apply(A1)
        for bc in bc_pressure:
            bc.apply(A2)

        solver1 = pde.LUSolver(A1)
        solver2 = pde.LUSolver(A2)
        solver3 = pde.LUSolver(A3)
        solver4 = pde.LUSolver(A4)
        def update_solution():
            # Step 1: Tentative velocity step
            b1 = pde.assemble(self.L1)
            for bc in bc_velocity:
                bc.apply(b1)
            solver1.solve(self.u_star_.vector(), b1)

            # Step 2: Pressure correction step
            b2 = pde.assemble(self.L2)
            for bc in bc_pressure:
                bc.apply(b2)
            solver2.solve(self.phi_.vector(), b2)

            # Step 3: Velocity correction step
            b3 = pde.assemble(self.L3)
            solver3.solve(self.u_.vector(), b3)
            
            # Step 4: Pressure update
            b4 = pde.assemble(self.L4)
            solver4.solve(self.p_.vector(), b4)
        self.update_solution = update_solution


def simulate_stokes(function_space, boundary_conditions):
    u, p = pde.TrialFunctions(function_space)
    v, q = pde.TestFunctions(function_space)
    a = (
        inner(nabla_grad(u), nabla_grad(v))*dx 
        + p*div(v)*dx + div(u)*q*dx
    )
    L = (
        dot(pde.Constant((0, 0)), v) * dx 
        + pde.Constant(0)*q * dx
    )

    soln = pde.Function(function_space)
    pde.solve(a == L, soln, bcs=boundary_conditions)
    return soln.split()
