import numpy as np
from scipy.integrate import simpson, quad
from mode_solver.flrw_native_riccaticpp_bundle.flrw_native_bundle.mode_solver_native import load_mode      #take new mode solver from riccaticpp for second order ODE's of similar form
import h5py
from itertools import product
from scipy.linalg import expm, solve
from functools import partial
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
from scipy.special import expit

#import physical input
from input import box_length_L, lambda_coupl, mass_probe_and_system, number_of_modes_per_direction, a, a_prime, a_2prime, compact_smearing_function, function_for_compact_coupling_region, function_to_obtain_mode, four_velocity, mu_sigma, mu_omega, t_grid

HERE = Path(__file__).resolve().parent

MODE_DIR = (
    HERE
    / "mode_solver"
    / "flrw_native_riccaticpp_bundle"
    / "flrw_native_bundle"
)

h5_path = MODE_DIR / "mode_solutions_native_v2.h5"
MIN_RUN = "quadratic_minimal_101"
NONMIN_RUN = "quadratic_non_minimal_101"

#parameters of physical model

L = box_length_L()  #to be chosen later (defines spatial volume)

N = number_of_modes_per_direction() #number of k in each direction

with h5py.File(h5_path, "r") as h5:
        obj = h5[f"runs/{MIN_RUN}/background/eta"]

        if not isinstance(obj, h5py.Dataset):
            raise TypeError("The eta path does not contain an HDF5 dataset")

        eta_values = np.asarray(obj[:], dtype=np.float64)

x = np.linspace(0, L, 100)
y = np.linspace(0, L, 100)
z = np.linspace(0, L, 100)

X, Y, Z = np.meshgrid(x, y, z, indexing="ij", sparse=True)


n_vectors = np.array(
                list(product(range(-N, N+1), repeat=3)),
                dtype=int
            )

momenta = (2.0 * np.pi / L) * n_vectors

k_x, k_y, k_z = momenta.T

number_of_modes = len(momenta)

m_probe, m_system = mass_probe_and_system()

lambda_coupling = lambda_coupl()

#smearing for coupling region
def rho_coupling(x, y, z, eta):
    return function_for_compact_coupling_region(eta, x, y, z)
#metric
def gmunu(eta):
    gmunu = np.array(
        [[- a(eta)**2, 0, 0, 0],
        [0, a(eta)**2, 0, 0],
        [0, 0, a(eta)**2, 0],
        [0, 0, 0, a(eta)**2]], dtype=np.float64
    )
    return gmunu

#velocity
def umu(eta):
    umu = np.asarray(four_velocity(eta), dtype=np.float64)
    return umu

def get_dataset(group: h5py.Group, path: str) -> h5py.Dataset:
    obj = group.get(path)

    if not isinstance(obj, h5py.Dataset):
        raise ValueError(f"{path!r} is missing or is not an HDF5 dataset")

    return obj


def load_run(run_name: str):
    with h5py.File(h5_path, "r") as h5:
        runs = h5.get("runs")

        if not isinstance(runs, h5py.Group):
            raise ValueError("'runs' is missing or is not an HDF5 group")

        run = runs.get(run_name)

        if not isinstance(run, h5py.Group):
            raise ValueError(
                f"Run {run_name!r} is missing or is not an HDF5 group"
            )

        k_data = get_dataset(run, "modes/k")
        eta_data = get_dataset(run, "background/eta")
        chi_data = get_dataset(run, "modes/chi")
        chi_prime_data = get_dataset(run, "modes/chi_prime")

        return (
            np.asarray(k_data[...], dtype=np.float64),
            np.asarray(eta_data[...], dtype=np.float64),
            np.asarray(chi_data[...], dtype=np.complex128),
            np.asarray(chi_prime_data[...], dtype=np.complex128),
        )

k_minimal_grid, eta_minimal_grid, chi_minimal_grid, chi_prime_minimal_grid = load_run("quadratic_minimal_101")
k_non_minimal_grid, eta_non_minimal_grid, chi_non_minimal_grid, chi_prime_non_minimal_grid = load_run("quadratic_non_minimal_101")

chi_minimal_interp = RegularGridInterpolator(
    (k_minimal_grid, eta_minimal_grid),
    chi_minimal_grid,
    bounds_error=True
)

chi_prime_minimal_interp = RegularGridInterpolator(
    (k_minimal_grid, eta_minimal_grid),
    chi_prime_minimal_grid,
    bounds_error=True
)

chi_non_minimal_interp = RegularGridInterpolator(
    (k_non_minimal_grid, eta_non_minimal_grid),
    chi_non_minimal_grid,
    bounds_error=True
)

chi_prime_non_minimal_interp = RegularGridInterpolator(
    (k_non_minimal_grid, eta_non_minimal_grid),
    chi_prime_non_minimal_grid,
    bounds_error=True
)

wronskian1 = (
    chi_minimal_grid * np.conj(chi_prime_minimal_grid)
    - chi_prime_minimal_grid * np.conj(chi_minimal_grid)
)

wronskian2 = (
    chi_non_minimal_grid * np.conj(chi_prime_non_minimal_grid)
    - chi_prime_non_minimal_grid * np.conj(chi_non_minimal_grid)
)

if not np.allclose(wronskian1, 1j, rtol=1e-7, atol=1e-9) or not np.allclose(wronskian2, 1j, rtol=1e-7, atol=1e-9):
    raise ValueError("Mode Wronskian normalization failed")

def chi(k, target_eta):                                     #Note to myself: target_eta has to be in the array eta, so check the grid with the actual mode solver
    return chi_minimal_interp((float(k), float(target_eta))).item()

def chi_prime(k, target_eta):
    return chi_prime_minimal_interp((float(k), float(target_eta))).item()

def chi_non_min(k, target_eta):
    return chi_non_minimal_interp((float(k), float(target_eta))).item()

def chi_prime_non_min(k, target_eta):
    return chi_prime_non_minimal_interp((float(k), float(target_eta))).item()

#frequencies of the field
def u(k_x, k_y, k_z, x, y, z, eta):
    u = 1/np.sqrt(L**3) * chi(np.sqrt(k_x**2 + k_y**2 + k_z**2), eta) / a(eta) * np.exp(1j * (k_x * x + k_y * y + k_z * z))
    return u

def u_partial(k_x, k_y, k_z, x, y, z, eta):
    k = np.sqrt(k_x**2 + k_y**2 + k_z**2)
    chi_value = chi(k, eta)
    chi_prime_value = chi_prime(k, eta)
    phase = np.exp(1j * (k_x*x + k_y*y + k_z*z))
    u = 1j * 1/np.sqrt(L**3) * chi_value / a(eta) * phase
    u_eta = 1/np.sqrt(L**3) * (chi_prime_value / a(eta) - chi_value * a_prime(eta)/(a(eta)**2) ) * phase
    partial = np.array([u_eta, u * k_x, u * k_y, u * k_z], dtype=np.complex128)
    return partial

def u_partial_eta(k_x, k_y, k_z, x, y, z, eta):
    u = 1/np.sqrt(L**3) * (chi_prime(np.sqrt(k_x**2 + k_y**2 + k_z**2), eta) / a(eta) - chi(np.sqrt(k_x**2 + k_y**2 + k_z**2), eta) * a_prime(eta)/(a(eta)**2) ) * np.exp(1j * (k_x * x + k_y * y + k_z * z))
    return u

def ubar(k_x, k_y, k_z, x, y, z, eta):       #lol
    return np.conj(
        u(k_x, k_y, k_z, x, y, z, eta)
        )

def ubar_partial(k_x, k_y, k_z, x, y, z, eta):       #lol
    return np.conj(
        u_partial(k_x, k_y, k_z, x, y, z, eta)
        )

#compact smearing function
def f(eta, x, y, z):
    result = compact_smearing_function(eta, x, y, z)
    return result

def g_inverse(eta):
    return np.diag([-1.0, 1.0, 1.0, 1.0]) / a(eta)**2

def energy_core(mode1, mode2, partial1, partial2, eta, smearing):
    velocity = umu(eta)
    g_cov = gmunu(eta)
    g_inv = g_inverse(eta)

    velocity_partial1 = sum(velocity[i] * partial1[i] for i in range(4))
    velocity_partial2 = sum(velocity[i] * partial2[i] for i in range(4))

    derivative_contraction = sum(g_inv[i, j] * partial1[i] * partial2[j] for i, j in product(range(4), repeat = 2))

    velocity_norm = sum(g_cov[i, j] * velocity[i] * velocity[j] for i, j in product(range(4), repeat  = 2))

    return a(eta)**4 * smearing * (velocity_partial1 * velocity_partial2 - 0.5 * velocity_norm * (derivative_contraction + m_probe**2 * mode1 * mode2))

def integrand_F_N(k1_x, k1_y, k1_z, k2_x, k2_y, k2_z, eta_n, X, Y, Z):

        mode1 = ubar(k1_x, k1_y, k1_z, X, Y, Z, eta_n)
        mode2 = u(k2_x, k2_y, k2_z, X, Y, Z, eta_n)

        partial1 = ubar_partial(k1_x, k1_y, k1_z, X, Y, Z, eta_n)
        partial2 = u_partial(k1_x, k1_y, k1_z, X, Y, Z, eta_n)

        return energy_core(mode1, mode2, partial1, partial2, eta_n, f(eta_n, X, Y, Z))


def integrand_G_N(k1_x, k1_y, k1_z, k2_x, k2_y, k2_z, eta_n, X, Y, Z):

        mode1 = u(k1_x, k1_y, k1_z, X, Y, Z, eta_n)
        mode2 = u(k2_x, k2_y, k2_z, X, Y, Z, eta_n)

        partial1 = u_partial(k1_x, k1_y, k1_z, X, Y, Z, eta_n)
        partial2 = u_partial(k1_x, k1_y, k1_z, X, Y, Z, eta_n)

        return energy_core(mode1, mode2, partial1, partial2, eta_n, f(eta_n, X, Y, Z))

def integrate_spatial(values, x, y, z):
    result = simpson(values, x = z, axis=2)
    result = simpson(result, x = y, axis=1)
    result = simpson(result, x = x, axis=0)
    return result

#helping functions for the energy density
def F_N_element(k_x, k_y, k_z, q_x, q_y, q_z):
    spatial_integrals = np.empty(len(eta_values), dtype=np.complex128)

    for n, eta_n in enumerate(eta_values):
        values = integrand_F_N(k_x, k_y, k_z, q_x, q_y, q_z, eta_n, X, Y, Z)

        spatial_integrals[n] = integrate_spatial(values, x, y, z)

    I = simpson(spatial_integrals, x = eta_values)

    return I

def G_N_element(k_x, k_y, k_z, q_x, q_y, q_z):
    spatial_integrals = np.empty(len(eta_values), dtype=np.complex128)
    
    for n, eta_n in enumerate(eta_values):
        values = integrand_G_N(k_x, k_y, k_z, q_x, q_y, q_z, eta_n, X, Y, Z)

        spatial_integrals[n] = integrate_spatial(values, x, y, z)

    I = simpson(spatial_integrals, x = eta_values)

    return I

def F_G_N():
    n_vectors = np.array(
            list(product(range(2 * N+1), repeat=3)),
            dtype=int
        )
    
    momenta = (2.0 * np.pi / L) * n_vectors

    number_of_modes = len(momenta)

    F = np.empty(
        (number_of_modes, number_of_modes),
        dtype=np.complex128
    )

    G = np.empty_like(F)

    for a, k_vec in enumerate(momenta):
        for b, q_vec in enumerate(momenta):
            F[a, b] = F_N_element(k_vec[0], k_vec[1], k_vec[2], q_vec[0], q_vec[1], q_vec[2])
            G[a, b] = G_N_element(k_vec[0], k_vec[1], k_vec[2], q_vec[0], q_vec[1], q_vec[2])

    return F, G
F, G = F_G_N()

def H_N():

    F_error = np.max(np.abs(F - F.conj().T))
    G_error = np.max(np.abs(G - G.T))

    if F_error > 1e-8:
        raise ValueError(f"F is not Hermitian: {F_error}")

    if G_error > 1e-8:
        raise ValueError(f"G is not symmetric: {G_error}")

    H = np.block([
        [(F + G).real, - (F + G).imag],
        [(F - G).imag, (F - G).real]
    ])

    return H

H = H_N()

M = (2 * N + 1)**3

zero = np.zeros((M, M), dtype=np.float64)
identity = np.eye(M, dtype=np.float64)

Omega = np.block([
        [zero, identity],
        [- identity, zero]
    ])

def S_N(t):
    return expm(- t * Omega @ H)

def C_N(t):
    ident = np.eye(2*M, dtype=np.float64)

    K = solve((S_N(t) + ident).T, (S_N(t) - ident).T).T

    C = - Omega @ K

    return C

#Field-Algebra and State input
def fundamental_solution(eta1, eta2, k_x, k_y, k_z):
    k = np.sqrt(k_x**2 + k_y**2 + k_z**2)
    chi_1 = chi(k, eta1)
    chi_2 = chi(k, eta2)
    return -1j * (chi_1 * np.conj(chi_2) - np.conj(chi_1) * chi_2)

def fundamental_solution_non_min(eta1, eta2, k_x, k_y, k_z):
    k = np.sqrt(k_x**2 + k_y**2 + k_z**2)
    chi_1 = chi_non_min(k, eta1)
    chi_2 = chi_non_min(k, eta2)
    return -1j * (chi_1 * np.conj(chi_2) - np.conj(chi_1) * chi_2)

def momentum_integration(values, momenta):
    result = simpson(values, x = momenta)
    return result

def retarded_greens_operator(target_eta, target_x, target_y, target_z, source, fundamental):
    eta_interval = eta_values[eta_values <= target_eta]
    time_integrand = np.empty(len(eta_interval), dtype=np.complex128)

    target_shape = np.broadcast_shapes(
        np.shape(target_x),
        np.shape(target_y),
        np.shape(target_z),
    )

    time_integrand = np.empty(
        (len(eta_interval),) + target_shape,
        dtype=np.complex128,
    )

    for n, eta_n in enumerate(eta_interval):
        source_values = source(eta_n, X, Y, Z)

        propagated_source = np.zeros(np.broadcast_shapes(X.shape, Y.shape, Z.shape), dtype=np.complex128)

        for k_vec in momenta:
            kx, ky, kz = k_vec

            source_k = integrate_spatial(source_values * np.exp(-1j * (kx * X + ky * Y + kz * Z)), x, y, z)

            

            propagated_source += (fundamental(target_eta, eta_n, kx, ky, kz) * source_k * np.exp(kx * target_x + ky * target_y + kz * target_z))

        values = (a(eta_n)**3 / (L**3 * a(target_eta)) * propagated_source * source_values)

        time_integrand[n] = integrate_spatial(values, x, y, z)
    
    return simpson(time_integrand, x = eta_interval, axis=0)

def advanced_greens_operator(target_eta, target_x, target_y, target_z, source, fundamental):
    eta_interval = eta_values[eta_values >= target_eta]
    time_integrand = np.empty(len(eta_interval), dtype=np.complex128)

    target_shape = np.broadcast_shapes(
        np.shape(target_x),
        np.shape(target_y),
        np.shape(target_z),
    )

    time_integrand = np.empty(
        (len(eta_interval),) + target_shape,
        dtype=np.complex128,
    )

    for n, eta_n in enumerate(eta_interval):
        source_values = source(eta_n, X, Y, Z)

        propagated_source = np.zeros(np.broadcast_shapes(X.shape, Y.shape, Z.shape), dtype=np.complex128)

        for k_vec in momenta:
            kx, ky, kz = k_vec

            source_k = integrate_spatial(source_values * np.exp(-1j * (kx * X + ky * Y + kz * Z)), x, y, z)

            propagated_source += (fundamental(target_eta, eta_n, kx, ky, kz) * source_k * np.exp(kx * target_x + ky * target_y + kz * target_z))

        values = (- a(eta_n)**3 / (L**3 * a(target_eta)) * propagated_source * source_values)

        time_integrand[n] = integrate_spatial(values, x, y, z)
    
    return simpson(time_integrand, x = eta_interval, axis=0)

def retarded_greens_operator_min(target_eta, target_x, target_y, target_z, source):
    return retarded_greens_operator(target_eta, target_x, target_y, target_z, source, fundamental_solution)

def advanced_greens_operator_min(target_eta, target_x, target_y, target_z, source):
    return advanced_greens_operator(target_eta, target_x, target_y, target_z, source, fundamental_solution)

def retarded_greens_operator_non_min(target_eta, target_x, target_y, target_z, source):
    return retarded_greens_operator(target_eta, target_x, target_y, target_z, source, fundamental_solution_non_min)

def advanced_greens_operator_non_min(target_eta, target_x, target_y, target_z, source):
    return advanced_greens_operator(target_eta, target_x, target_y, target_z, source, fundamental_solution_non_min)

def causal_propagator_min(target_eta, target_x, target_y, target_z, source):
    causl_propagator = retarded_greens_operator_min(target_eta, target_x, target_y, target_z, source) - advanced_greens_operator_min(target_eta, target_x, target_y, target_z, source)
    return causl_propagator

def causal_propagator_non_min(target_eta, target_x, target_y, target_z, source):
    causl_propagator = retarded_greens_operator_non_min(target_eta, target_x, target_y, target_z, source) - advanced_greens_operator_non_min(target_eta, target_x, target_y, target_z, source)
    return causl_propagator

def symplectic_min(l, o):
    spatial_integrals = np.empty(len(eta_values), dtype=np.complex128)

    for n, eta_n in enumerate(eta_values):
        values =  a(eta_n)**4 * causal_propagator_min(eta_n, X, Y, Z, o) * l(eta_n, X, Y, Z)

        spatial_integrals[n] = integrate_spatial(values, x, y, z)

    I = simpson(spatial_integrals, x = eta_values)
    
    return I

def symplectic_non_min(l, o):
    spatial_integrals = np.empty(len(eta_values), dtype=np.complex128)

    for n, eta_n in enumerate(eta_values):
        values =  a(eta_n)**4 * causal_propagator_non_min(eta_n, X, Y, Z, o) * l(eta_n, X, Y, Z)

        spatial_integrals[n] = integrate_spatial(values, x, y, z)

    I = simpson(spatial_integrals, x = eta_values)
    
    return I

#helping function to obtain smearing function for Phi to obtain certain a_i, a^dager_i
def rho(eta, x, y, z):
    rho_, rho_partial_, rho_2partial_ = function_to_obtain_mode(eta, x, y, z)
    return np.asarray(rho_, dtype=np.float64)

def rho_partial(eta, x, y, z):
    rho_, rho_partial_, rho_2partial_ = function_to_obtain_mode(eta, x, y, z)
    return np.asarray(rho_partial_, dtype=np.float64)

def rho_2partial(eta, x, y, z):
    rho_, rho_partial_, rho_2partial_ = function_to_obtain_mode(eta, x, y, z)
    return np.asarray(rho_2partial_, dtype=np.float64)

def g_i(eta, x, y, z, k_x, k_y, k_z):
    g_inv = g_inverse(eta)
    grad_rho = rho_partial(eta, x, y, z)
    hessian_rho = rho_2partial(eta, x, y, z)
    
    mode = ubar(k_x, k_y, k_z, x, y, z, eta)
    mode_partial = ubar_partial(k_x, k_y, k_z, x, y, z, eta)

    box_rho = (- hessian_rho[0] - 2 * a_prime(eta)/ a(eta) * grad_rho[0] + sum(hessian_rho[i] for i in range(1, 4))) / a(eta)**2
    cross_term = sum(g_inv[i, j] * grad_rho[i] * mode_partial[j] for i, j in product(range(4), repeat=2))
    return 1j * (box_rho * mode + 2 * cross_term)

def gbar_i(eta, x, y, z, k_x, k_y, k_z):
    return np.conj(g_i(eta, x, y, z, k_x, k_y, k_z))

def c_n():
    Gn = [partial(g_i, k_x = k_vec[0], k_y = k_vec[1], k_z = k_vec[2]) for k_vec in momenta]
    GBar = [partial(gbar_i, k_x = k_vec[0], k_y = k_vec[1], k_z = k_vec[2]) for k_vec in momenta]

    c = - sum((F[i][j] * (mu_sigma(partial(causal_propagator_min, source = GBar[i]), partial(causal_propagator_min, source = Gn[j])) + 1j/2 * symplectic_min(GBar[i], Gn[j])) 
                + 1/2 * G[i][j] * (mu_sigma(partial(causal_propagator_min, source = Gn[i]), partial(causal_propagator_min, source = Gn[j])) + 1j/2 * symplectic_min(Gn[i], Gn[j]))
                + 1/2 * np.conj(G[j][i]) * (mu_sigma(partial(causal_propagator_min, source = GBar[i]), partial(causal_propagator_min, source = GBar[j])) + 1j/2 * symplectic_min(GBar[i], GBar[j]))
               ) for i, j in product(range(number_of_modes), repeat=2))

    return c

c = c_n()

def f_j_minus(eta, x, y, z, k_x, k_y, k_z):
    f_j = - lambda_coupling * retarded_greens_operator_min(eta, x, y, z, partial(g_i, k_x = k_x, k_y = k_y, k_z = k_z)) * rho_coupling(x, y, z, eta)
    return f_j

def h_j_minus(eta, x, y, z, k_x, k_y, k_z):
    h_j = g_i(eta, x, y, z, k_x, k_y, k_z)
    return h_j

def f_tilde_j_minus(eta, x, y, z, k_x, k_y, k_z):
    f_tilde_j = - lambda_coupling * retarded_greens_operator_min(eta, x, y, z, partial(gbar_i, k_x = k_x, k_y = k_y, k_z = k_z)) * rho_coupling(x, y, z, eta)
    return f_tilde_j

def h_tilde_j_minus(eta, x, y, z, k_x, k_y, k_z):
    h_tilde_j = gbar_i(eta, x, y, z, k_x, k_y, k_z)
    return h_tilde_j


def N_ij(k_x, k_y, k_z, q_x, q_y, q_z):
    N_ij = (mu_omega(partial(causal_propagator_non_min, source = partial(f_tilde_j_minus, k_x = k_x, k_y = k_y, k_z = k_z)), partial(causal_propagator_non_min, source = partial(f_j_minus, k_x = q_x, k_y = q_y, k_z = q_z)))
    + mu_sigma(partial(causal_propagator_min, source = partial(h_tilde_j_minus, k_x = k_x, k_y = k_y, k_z = k_z)), partial(causal_propagator_min, source = partial(h_j_minus, k_x = q_x, k_y = q_y, k_z = q_z))) 
    + 1j/2 * symplectic_non_min(partial(f_tilde_j_minus, k_x = k_x, k_y = k_y, k_z = k_z), partial(f_j_minus, k_x = q_x, k_y = q_y, k_z = q_z))
    + 1j/2 * symplectic_min(partial(h_tilde_j_minus, k_x = k_x, k_y = k_y, k_z = k_z), partial(h_j_minus, k_x = q_x, k_y = q_y, k_z = q_z)))
    return N_ij

def M_ij(k_x, k_y, k_z, q_x, q_y, q_z):
    M_ij = (mu_omega(partial(causal_propagator_non_min, source = partial(f_j_minus, k_x = k_x, k_y = k_y, k_z = k_z)), partial(causal_propagator_non_min, source = partial(f_j_minus, k_x = q_x, k_y = q_y, k_z = q_z)))
        + mu_sigma(partial(causal_propagator_min, source = partial(h_j_minus, k_x = k_x, k_y = k_y, k_z = k_z)), partial(causal_propagator_min, source = partial(h_j_minus, k_x = q_x, k_y = q_y, k_z = q_z))) 
        + 1j/2 * symplectic_non_min(partial(f_j_minus, k_x = k_x, k_y = k_y, k_z = k_z), partial(f_j_minus, k_x = q_x, k_y = q_y, k_z = q_z))
        + 1j/2 * symplectic_min(partial(h_j_minus, k_x = k_x, k_y = k_y, k_z = k_z), partial(h_j_minus, k_x = q_x, k_y = q_y, k_z = q_z)))
    return M_ij

def N_M():
    Nn = np.empty(
        (len(momenta), len(momenta)),
        dtype=np.complex128
    )

    M = np.empty_like(Nn)

    for i, k_vec in enumerate(momenta):
        for j, q_vec in enumerate(momenta):
            Nn[i][j] = N_ij(k_vec[0], k_vec[1], k_vec[2], q_vec[0], q_vec[1], q_vec[2])
            M[i][j] = M_ij(k_vec[0], k_vec[1], k_vec[2], q_vec[0], q_vec[1], q_vec[2])

    if not np.allclose(Nn, Nn.conj().T, atol=1e-8):
        raise ValueError("N is not Hermitian")

    if not np.allclose(M, M.T, atol=1e-8):
        raise ValueError("M is not symmetric")

    return Nn, M

def V_N():
    N, M = N_M()

    id = np.eye(len(N), dtype=np.float64)

    V_pp = 1/2 * (id + N + N.T - M - M.conj().T)
    V_qq = 1/2 * (id + N + N.T + M + M.conj().T)
    V_qp = - 1j/2 * (N - N.T + M - M.conj().T)

    V = np.block([
        [V_qq, V_qp],
        [V_qp.T, V_pp]])

    return V

V = V_N()

minimum = np.linalg.eigvalsh(V + 0.5j * Omega).min()

if minimum < -1e-8:
    raise ValueError("Covariance matrix is not physical")

def helping_function(t):
    res = 0.0
    if t > 0:
        res = np.exp(- 1/t)
    return res

def charac(epsilon, delta, s):
    if delta <= 0:
        raise ValueError("delta must be positive")

    if s <= epsilon:
        return 0.0

    if s >= epsilon + delta:
        return 1.0
    
    erg = helping_function(s - epsilon)/(helping_function(s - epsilon) + helping_function(epsilon + delta - s))
    return erg

def charac_prime(epsilon, delta, s):
    if s <= epsilon or s >= epsilon + delta:
        return 0.0

    proizlaziti = (1/(s - epsilon)**2 * charac(epsilon, delta, s) - charac(epsilon, delta, s) /(helping_function(s - epsilon) + helping_function(epsilon + delta - s)) 
             * (1/(s - epsilon)**2 * helping_function(s - epsilon) - 1/(epsilon + delta - s)**2 * helping_function(epsilon + delta - s)))
    return proizlaziti

def F_delta_integrand(s, epsilon, delta, t):
    risultato = charac_prime(epsilon, delta, s) * np.exp(- 1j * t * s)
    return risultato

def F_delta(t, delta, epsilon):
    value, err = quad(F_delta_integrand, epsilon, epsilon + delta, args=(epsilon, delta, t), complex_func=True)
    return value

def probability(epsilon, delta):
    identt = np.eye(2 * M, dtype=np.float64)

    def probability_integrand(t, epsilon, delta):
        if t == 0:
            mean_A = np.real(c + 0.5 * np.trace(H @ V))
            return float(mean_A - epsilon - delta / 2)
        S = S_N(t)
        jieguo = 1/t * (F_delta(t, delta, epsilon) * np.exp(1j * t * c) 
                * (np.linalg.det((identt + S -  2 * 1j * V @ Omega @ (S - identt))/2))**(- 1/2)).imag
        return jieguo

    t_values = t_grid(epsilon, delta)

    probability_slices = np.empty(len(t_values), dtype=np.float64)

    for n, t_n in enumerate(t_values):
        probability_slices[n] = probability_integrand(t_n, epsilon, delta)
        

    integral = simpson(probability_slices, x = t_values)
    kekka = 0.5 + integral / np.pi
    return kekka
