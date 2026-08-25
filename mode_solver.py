import h5py
import numpy as np
from scipy.integrate import solve_ivp

#Scaling factor and it's second derivative for FLWR metric

def a(eta):
    return eta**2

def a_double_prime(eta):
    return 2

#Frequency in Mode-Equation

def omega(k, x, m, eta):
    omega = (
        k**2 
        + m**2 * a(eta)**2 
        - (1- 6*x)* a_double_prime(eta)/a(eta)
    )
    return omega

#ODE input

def rhs(eta, y, k, x, m):
    chi = y[0]
    chi_prime = y[1]
    return np.array([
        chi_prime,
        - omega(k, x, m, eta) * chi
    ], dtype=np.complex128)

#Initial conditions for ODE

def initial_conditions(k, m, x, eta_initial):
    chi_initial = k                                 #I will choose realistic functions later, maybe plane waves are good. 
    chi_prime_initial = k
    return np.array([
        chi_initial,
        chi_prime_initial
    ], dtype=np.complex128)

#Solver for ODE

def solve(k, x, m, eta_initial, eta_final, rtol, atol, y0, method):
    y0 = np.asarray(y0, dtype=np.complex128)
    solution = solve_ivp(
        rhs,
        (eta_initial, eta_final),
        y0,
        args=(k, x, m),
        method=method,
        dense_output=True,
        rtol=rtol,
        atol=atol
    )

    return solution

#Structure and Write solutions into mode_solutions.h5

def save_sol(filename, run_name, k_values, eta_values, m, x, a_model_name, method, rtol, atol, meth):
    with h5py.File(filename, "a") as h5:
        runs = h5.require_group("runs")

        if run_name in runs:
            raise ValueError(
                f"Run {run_name} already exists"
            )

        run = runs.create_group(run_name)

        run.attrs["mass"] = m
        run.attrs["Xi"] = x
        run.attrs["a_model"] = a_model_name
        run.attrs["method"] = meth
        run.attrs["rtol"] = rtol
        run.attrs["atol"] = atol

        number_of_k = len(k_values)
        number_of_eta = len(eta_values)

        k_values = np.asarray(
            k_values,
            dtype=np.float64
        )
        eta_values = np.asarray(
            eta_values,
            dtype=np.float64
        )

        background = run.create_group("background")

        background.create_dataset(
            "eta",
            data=eta_values
        )

        background.create_dataset(
            "a",
            data=a(eta_values)
        )

        modes = run.create_group("modes")

        k_dataset = modes.create_dataset(
            "k",
            shape=(number_of_k, ),
            dtype=np.float64
        )

        chi_dataset = modes.create_dataset(
            "chi",
            shape=(number_of_k, number_of_eta),
            dtype=np.complex128,
            chunks=(1, min(8192, number_of_eta)),
            compression="lzf",
            shuffle=True,
            fletcher32=True
        )

        chi_prime_dataset = modes.create_dataset(
            "chi_prime",
            shape=(number_of_k, number_of_eta),
            dtype=np.complex128,
            chunks=(1, min(8192, number_of_eta)),
            compression="lzf",
            shuffle=True,
            fletcher32=True
        )

        completed_dataset = modes.create_dataset(
            "completed",
            shape=(number_of_k,),
            dtype=np.bool_
        )

        completed_dataset[:] = False

        for i, k in enumerate(k_values):
            y0 = initial_conditions(k, m, x, eta_values[0])

            solution = solve(k, x, m, eta_values[0], eta_values[-1], rtol, atol, y0, meth)

            if not solution.success():
                raise RuntimeError(
                    f"Solver failed for {k}"
                    f"{solution.message}"
                )

            solution_values = solution.sol(eta_values)

            k_dataset[i] = k
            chi_dataset[i, :] = solution_values[0]
            chi_prime_dataset[i, :] = solution_values[1]
            completed_dataset[i] = True

            h5.flush()

    return None

