import argparse
import pandas as pd
import fenics as pde
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def estimate_error_bound(result_df, error_column):
    h = result_df['h'].values
    log_h = np.log(h)
    err = result_df[error_column].values
    log_err = np.log(err)
    
    def loss_function(x):
        error = np.log(x[0]) + x[1]*log_h - log_err
        error[error < 0] *= 1e8
        return np.sum(error**2)
    C, power = optimize.minimize(loss_function, [5, 1], method='Nelder-Mead').x
    return C, power


def solve_petrov_galerkin(resolution, mu_value, beta_value=0.5):
    from fenics import (
        dx, inner, grad
    )
    mesh = pde.UnitIntervalMesh(resolution)
    V = pde.FunctionSpace(mesh, "CG", 1)
    u = pde.TrialFunction(V)
    v_galerkin = pde.TestFunction(V)

    mu = pde.Constant(mu_value)
    beta = pde.Constant(beta_value)

    f = pde.Constant(0)
    h_value = mesh.hmin()
    h = pde.Constant(h_value)
    v = v_galerkin + beta*h*v_galerkin.dx(0)

    a = (u.dx(0)*v + mu*u.dx(0)*v.dx(0))*dx
    L = f*(v - beta*h*v.dx(0)) * dx

    bcs = [
        pde.DirichletBC(V, 0, "near(x[0], 0)"),
        pde.DirichletBC(V, 1, "near(x[0], 1)"),
    ]

    u_true = pde.Expression(
        "(exp(x[0]/(2*mu))*sinh(x[0]/(2*mu))) /"
        "(exp(1/(2*mu))*sinh(1/(2*mu)))",
        mu=mu,
        degree=7,
    )
    
    u_ = pde.Function(V)
    pde.solve(a == L, u_, bcs=bcs)

    results = {
        r"$\mu$": mu_value,
        "Resolution": resolution,
        "h": h_value,
        "Error (L2)": pde.errornorm(u_true, u_, "L2"),
        "Error (H1)": pde.errornorm(u_true, u_, "H1"),
        "Scheme": "Petrov-Galerkin",
        }

    return u_, results


def solve_galerkin(resolution, mu_value):
    from fenics import (
        dx, inner, grad
    )
    mesh = pde.UnitIntervalMesh(resolution)
    V = pde.FunctionSpace(mesh, "CG", 1)
    u = pde.TrialFunction(V)
    v = pde.TestFunction(V)

    f = pde.Constant(0)
    h = mesh.hmin()
    mu = pde.Constant(mu_value)

    a = (mu*inner(grad(u), grad(v))*dx + u.dx(0)*v*dx)
    L = f*v*dx

    bcs = [
        pde.DirichletBC(V, 0, "near(x[0], 0)"),
        pde.DirichletBC(V, 1, "near(x[0], 1)"),
    ]

    u_true = pde.Expression(
        "(exp(x[0]/(2*mu))*sinh(x[0]/(2*mu))) /"
        "(exp(1/(2*mu))*sinh(1/(2*mu)))",
        mu=mu,
        degree=7,
    )
    
    u_ = pde.Function(V)
    pde.solve(a == L, u_, bcs=bcs)

    results = {
            "Scheme": "Galerkin",
        r"$\mu$": mu_value,
        "Resolution": resolution,
        "h": h,
        "Error (H1)": pde.errornorm(u_true, u_, "H1"),
        "Error (L2)": pde.errornorm(u_true, u_, "L2"),
        }

    return u_, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--format', type=str, default="text", help="text or latex")
    args = parser.parse_args()

    if args.format.lower() not in {'text', 'latex'}:
        raise ValueError("Format argument must be equal to 'text' or 'latex'")

    results = []
    for mu_value in [1, 0.3, 0.1]:
        for resolution in [8, 16, 32, 64]:
            u_g, results_galerkin = solve_galerkin(resolution, mu_value)
            results.append(results_galerkin)
            u_pg, results_petrov_galerkin = solve_petrov_galerkin(resolution, mu_value)
            results.append(results_petrov_galerkin)

    results = (
        pd.DataFrame(results)
        .set_index(["Scheme", r"$\mu$", "Resolution"])
        .sort_index()
    )

    coefficients = []
    for setup, result in results.groupby(level=[0, 1]):
        scheme, mu = setup

        C_alpha, alpha = estimate_error_bound(result, 'Error (H1)')
        C_beta, beta = estimate_error_bound(result, 'Error (L2)')
        coefficients.append({'Scheme': scheme, r'$\mu$': mu, r'$C_\alpha$': C_alpha, r'$\alpha$': alpha, r'$C_\beta$': C_beta, r'$\beta$': beta})

        h = results.loc[setup, 'h'].values
        results.loc[setup, 'Estimated error (H1)'] = C_alpha * (h**alpha)
        results.loc[setup, 'Estimated error (L2)'] = C_beta * (h**beta)
    coefficients = (
        pd.DataFrame(coefficients)
        .set_index(['Scheme', r'$\mu$'])
        .sort_index()
    )
    
    mispred_h1 = (results['Estimated error (H1)'] - results['Error (H1)']) / results['Error (H1)']

    mispred_l2 = (results['Estimated error (L2)'] - results['Error (L2)']) / results['Error (L2)']

    results = results[['Error (H1)', 'Estimated error (H1)', 'Error (L2)', 'Estimated error (L2)']]
    
    if args.format.lower() == "latex":
        print(coefficients.to_latex(float_format=lambda x: f"{x:.2f}", escape=False))
        print(results.to_latex(float_format=lambda x: f"{x:.2f}", escape=False))
    else:
        print(coefficients)
        print(results)

    print("The maximal misprediction was:")
    print(f"  H1: {mispred_h1.max()} with {mispred_h1.idxmax()}")
    print(f"  L2: {mispred_l2.max()} with {mispred_l2.idxmax()}")

    results['h'] = 1/results.reset_index()['Resolution'].values
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    sns.lineplot(x='h', y="Error (L2)", hue=r"$\mu$", style="Scheme", data=results.reset_index(), palette="Set1")
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.savefig("Convergence_plots_l2.pdf")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    sns.lineplot(x='h', y="Error (H1)", hue=r"$\mu$", style="Scheme", data=results.reset_index(), palette="Set1")
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.savefig("Convergence_plots_h2.pdf")
