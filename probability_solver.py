import numpy as np
from scipy.integrate import simpson, quad
from mode_solver.flrw_native_riccaticpp_bundle.flrw_native_bundle.mode_solver_native import load_mode      #take new mode solver from riccaticpp for second order ODE's of similar form
import h5py
from itertools import product
from scipy.linalg import expm, solve
from functools import partial


#import physical input
from input import box_length_L, lambda_coupl, mass_probe_and_system, number_of_modes_per_direction, a, a_prime, a_2prime, compact_smearing_function, function_for_compact_coupling_region, function_to_obtain_mode, four_velocity, mu_sigma, mu_omega


#parameters of physical model

L = box_length_L()  #to be chosen later (defines spatial volume)

N = number_of_modes_per_direction() #number of k in each direction

with h5py.File("mode_solver/mode_solutions_native.h5", "r") as h5:
        obj = h5["runs/quadratic_test_002/background/eta"]

        if not isinstance(obj, h5py.Dataset):
            raise TypeError("The eta path does not contain an HDF5 dataset")

        eta_values = np.asarray(obj[:], dtype=np.float64)

x = np.linspace(0, L, 100)
y = np.linspace(0, L, 100)
z = np.linspace(0, L, 100)

X, Y, Z = np.meshgrid(x, y, z, indexing="ij", sparse=True)


n_vectors = np.array(
                list(product(range(N+1), repeat=3)),
                dtype=int
            )
        
momenta = (2.0 * np.pi / L) * n_vectors

k_x, k_y, k_z = momenta[0], momenta[1], momenta[2]

K_x, K_y, K_z = np.meshgrid(k_x, k_y, k_z, indexing="ij", sparse=True)

m_probe, m_system = mass_probe_and_system()

lambda_coupling = lambda_coupl()

#smearing for coupling region
def rho_coupling(x, y, z, eta):
    rho_coupl = function_for_compact_coupling_region(eta, x, y, z)
    return rho_coupl

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

def chi(k, target_eta):                                     #Note to myself: target_eta has to be in the array eta, so check the grid with the actual mode solver
    k_found, eta_raw, chi_k_raw, chi_prime_k_raw = load_mode(
        filename = "mode_solutions_native.h5",
        run_name = "quadratic_minimal_101",
        k = k
    )

    eta = np.asarray(eta_raw, dtype=np.float64)
    chi_k = np.asarray(chi_k_raw, dtype=np.complex128)
    chi_prime_k = np.asarray(chi_prime_k_raw, dtype=np.complex128)

    i = np.searchsorted(eta, target_eta)
    return chi_k[i]

def chi_prime(k, target_eta):
    k_found, eta_raw, chi_k_raw, chi_prime_k_raw = load_mode(
            filename = "mode_solutions_native.h5",
            run_name = "quadratic_minimal_101",
            k = k
        )

    eta = np.asarray(eta_raw, dtype=np.float64)
    chi_k = np.asarray(chi_k_raw, dtype=np.complex128)
    chi_prime_k = np.asarray(chi_prime_k_raw, dtype=np.complex128)
    
    i = np.searchsorted(eta, target_eta)
    return chi_prime_k[i]

def chi_non_min(k, target_eta):
    k_found, eta_raw, chi_k_raw, chi_prime_k_raw = load_mode(
        filename = "mode_solutions_native_non_minimal.h5",
        run_name = "quadratic_non_minimal_101",
        k = k
    )

    eta = np.asarray(eta_raw, dtype=np.float64)
    chi_k = np.asarray(chi_k_raw, dtype=np.complex128)
    chi_prime_k = np.asarray(chi_prime_k_raw, dtype=np.complex128)

    i = np.searchsorted(eta, target_eta)
    return chi_k[i]

def chi_prime_non_min(k, target_eta):
    k_found, eta_raw, chi_k_raw, chi_prime_k_raw = load_mode(
        filename = "mode_solutions_native_non_minimal.h5",
        run_name = "quadratic_non_minimal_101",
        k = k
    )

    eta = np.asarray(eta_raw, dtype=np.float64)
    chi_k = np.asarray(chi_k_raw, dtype=np.complex128)
    chi_prime_k = np.asarray(chi_prime_k_raw, dtype=np.complex128)

    i = np.searchsorted(eta, target_eta)
    return chi_prime_k[i]

#frequencies of the field
def u(k_x, k_y, k_z, x, y, z, eta):
    u = 1/np.sqrt(L**3) * chi(np.sqrt(k_x**2 + k_y**2 + k_z**2), eta) / a(eta) * np.exp(1j * (k_x * x + k_y * y + k_z * z))
    return u

def u_partial(k_x, k_y, k_z, x, y, z, eta):
    u = 1j * 1/np.sqrt(L**3) * chi(np.sqrt(k_x**2 + k_y**2 + k_z**2), eta) / a(eta) * np.exp(1j * (k_x * x + k_y * y + k_z * z))
    u_eta = 1/np.sqrt(L**3) * (chi_prime(np.sqrt(k_x**2 + k_y**2 + k_z**2), eta) / a(eta) - chi(np.sqrt(k_x**2 + k_y**2 + k_z**2), eta) * a_prime(eta)/(a(eta)**2) ) * np.exp(1j * (k_x * x + k_y * y + k_z * z))
    partial = np.array([u_eta, u * k_x, u * k_y, u * k_z], dtype=np.complex128)
    return partial

def u_partial_eta(k_x, k_y, k_z, x, y, z, eta):
    u = 1/np.sqrt(L**3) * (chi_prime(np.sqrt(k_x**2 + k_y**2 + k_z**2), eta) / a(eta) - chi(np.sqrt(k_x**2 + k_y**2 + k_z**2), eta) * a_prime(eta)/(a(eta)**2) ) * np.exp(1j * (k_x * x + k_y * y + k_z * z))
    return u

def ubar(k_x, k_y, k_z, x, y, z, eta):       #lol
    u = 1/np.sqrt(L**3) * (chi(np.sqrt(k_x**2 + k_y**2 + k_z**2), eta).real - 1j * chi(np.sqrt(k_x**2 + k_y**2 + k_z**2), eta).imag)  / a(eta) * np.exp(1j * (k_x * x + k_y * y + k_z * z))
    return u

def ubar_partial(k_x, k_y, k_z, x, y, z, eta):       #lol
    u = - 1j  * 1/np.sqrt(L**3) * (chi(np.sqrt(k_x**2 + k_y**2 + k_z**2), eta).real - 1j * chi(np.sqrt(k_x**2 + k_y**2 + k_z**2), eta).imag)  / a(eta) * np.exp(- 1j * (k_x * x + k_y * y + k_z * z))
    u_eta = 1/np.sqrt(L**3) * ((chi_prime(np.sqrt(k_x**2 + k_y**2 + k_z**2), eta).real - 1j * chi_prime(np.sqrt(k_x**2 + k_y**2 + k_z**2), eta).imag)/ a(eta) - (chi(np.sqrt(k_x**2 + k_y**2 + k_z**2), eta).real - 1j * chi(np.sqrt(k_x**2 + k_y**2 + k_z**2), eta)) * a_prime(eta)/(a(eta)**2) ) * np.exp(1j * (k_x * x + k_y * y + k_z * z))
    partial = np.array([u_eta, u * k_x, u * k_y, u * k_z], dtype=np.complex128)
    return partial

#compact smearing function
def f(eta, x, y, z):
    result = compact_smearing_function(eta, x, y, z)
    return result

def integrand_F_N(k1_x, k1_y, k1_z, k2_x, k2_y, k2_z, eta_n, X, Y, Z):
        #smearing function
        f_smearing = f(eta_n, X, Y, Z)

        #velocity
        vel = umu(eta_n)

        #modes
        ubar1 = ubar(k1_x, k1_y, k1_z, X, Y, Z, eta_n)
        u2 = u(k2_x, k2_y, k2_z, X, Y, Z, eta_n)
        ubar1_partial = ubar_partial(k1_x, k1_y, k1_z, X, Y, Z, eta_n)
        u2_partial = u_partial(k1_x, k1_y, k1_z, X, Y, Z, eta_n)

        a = 0
        b = 0
        c = 0
        d = 0
        e = 0

        for i in range(0, 4):
            a += u2_partial[i] * vel[i]
            b += ubar1_partial[i] * vel[i]
            c += u2_partial[i] * ubar1_partial[i]

            for j in range(0, 4):
                e += gmunu(eta_n)[i][j] * vel[i] * vel[j]
        
        d += u2 * ubar1

        return f_smearing * (a * b - 1/2 * e * (c + m_probe**2 * d))

def integrand_G_N(k1_x, k1_y, k1_z, k2_x, k2_y, k2_z, eta_n, X, Y, Z):
        #smearing function
        f_smearing = f(eta_n, X, Y, Z)

        #velocity
        vel = umu(eta_n)

        #modes
        ubar1 = ubar(k1_x, k1_y, k1_z, X, Y, Z, eta_n)
        u2 = u(k2_x, k2_y, k2_z, X, Y, Z, eta_n)
        ubar1_partial = u_partial(k1_x, k1_y, k1_z, X, Y, Z, eta_n)
        u2_partial = u_partial(k1_x, k1_y, k1_z, X, Y, Z, eta_n)

        a = 0
        b = 0
        c = 0
        d = 0
        e = 0

        for i in range(0, 4):
            a += u2_partial[i] * vel[i]
            b += ubar1_partial[i] * vel[i]
            c += u2_partial[i] * ubar1_partial[i]
            d += u2 * ubar1

            for j in range(0, 4):
                e += gmunu(eta_n)[i][j] * vel[i] * vel[j]

        return f_smearing * (a * b - 1/2 * e * (c + m_probe**2 * d))

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
            list(product(range(N+1), repeat=3)),
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

def H_N():
    F, G = F_G_N()

    H = np.block([
        [(F + G).real, - (F + G).imag],
        [(F - G).imag, (F - G).real]
    ])

    return H

def S_N(t):
    H = H_N()

    M = (N + 1)**3

    zero = np.zeros((M, M), dtype=np.float64)
    identity = np.eye(M, dtype=np.float64)

    Omega = np.block([
        [zero, identity],
        [- identity, zero]
    ])
    
    S = expm(t * Omega @ H)

    return S

def C_N(t):

    M = (N + 1)**3
    
    zero = np.zeros((M, M), dtype=np.float64)
    identity = np.eye(M, dtype=np.float64)
    ident = np.eye(2*M, dtype=np.float64)

    Omega = np.block([
        [zero, identity],
        [- identity, zero]
    ])

    K = solve((S_N(t) + ident).T, (S_N(t) - ident).T).T

    C = - Omega @ K

    return C

#Field-Algebra and State input
def fundamental_solution(eta1, eta2, k_x, k_y, k_z):
    k = np.sqrt(k_x**2 + k_y**2 + k_z**2)
    S = - 1j * (chi(k, eta2) * (chi(k, eta1).real - 1j * chi(k, eta1).imag) - (chi(k, eta2).real - 1j * chi(k, eta2).imag) * chi(k, eta1))
    return S

def momentum_integration(values, k_x, k_y, k_z):
    result = simpson(values, x = k_z, axis=2)
    result = simpson(values, x = k_y, axis=1)
    result = simpson(values, x = k_x, axis=0)
    return result

def greens_operator_min(eta, target_x, target_y, target_z, o):
    momentum_integrals = np.empty((len(eta_values), len(X), len(Y), len(Z)), dtype=np.complex128)

    for n, eta_n in enumerate(eta_values):
        for n_x, x_n in enumerate(X):
            for n_y, y_n in enumerate(Y):
                for n_z, z_n in enumerate(Z):
                    values = a(eta_n)**3 * np.exp(1j * (K_x * (x_n - target_x) + K_y * (y_n - target_y) + K_z * (z_n - target_z))) * fundamental_solution(eta, eta_n, momenta[0], momenta[1], momenta[2]) * o(eta_n, x_n, y_n, z_n)

                    momentum_integrals[n, n_x, n_y, n_z] = momentum_integration(values, k_x, k_y, k_z)

                result = simpson(momentum_integrals, x = X)
            result = simpson(result, x = Y)
        result = simpson(result, x = Z)

    idx = np.where(eta_values == eta)[0]

    eta_interval = eta_values[idx:]
    
    result = simpson(result, x = eta_interval)

    return result

def causal_propagator_min(eta, target_x, target_y, target_z, o):
    causl_propagator = 2 * 1/a(eta) * greens_operator_min(eta, target_x, target_y, target_z, o)
    return causl_propagator

def symplectic_min(l, o):
    spatial_integrals = np.empty(len(eta_values), dtype=np.complex128)

    for n, eta_n in enumerate(eta_values):
        values =  2 / a(eta_n) * greens_operator_min(eta_n, X, Y, Z, o) * l(eta_n, X, Y, Z)

        spatial_integrals[n] = integrate_spatial(values, x, y, z)

    I = simpson(spatial_integrals, x = eta_values)
    
    return I

def greens_operator_non_min(eta, target_x, target_y, target_z, o):
    momentum_integrals = np.empty((len(eta_values), len(X), len(Y), len(Z)), dtype=np.complex128)

    for n, eta_n in enumerate(eta_values):
        for n_x, x_n in enumerate(X):
            for n_y, y_n in enumerate(Y):
                for n_z, z_n in enumerate(Z):
                    values = a(eta_n)**3 * np.exp(1j * (K_x * (x_n - target_x) + K_y * (y_n - target_y) + K_z * (z_n - target_z))) * fundamental_solution(eta, eta_n, momenta[0], momenta[1], momenta[2]) * o(eta_n, x_n, y_n, z_n)

                    momentum_integrals[n, n_x, n_y, n_z] = momentum_integration(values, k_x, k_y, k_z)

                result = simpson(momentum_integrals, x = X)
            result = simpson(result, x = Y)
        result = simpson(result, x = Z)

    idx = np.where(eta_values == eta)[0]

    eta_interval = eta_values[idx:]
    
    result = simpson(result, x = eta_interval)

    return result

def causal_propagator_non_min(eta, target_x, target_y, target_z, o):
    causl_propagator = 2 * 1/a(eta) * greens_operator_non_min(eta, target_x, target_y, target_z, o)
    return causl_propagator

def symplectic_non_min(l, o):
    spatial_integrals = np.empty(len(eta_values), dtype=np.complex128)

    for n, eta_n in enumerate(eta_values):
        values =  2 / a(eta_n) * greens_operator_non_min(eta_n, X, Y, Z, o) * l(eta_n, X, Y, Z)

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
    g = 1j * (sum(gmunu(eta)[i][i] * rho_partial(eta, x, y, z)[i] * ubar(k_x, k_y, k_z, x, y, z, eta) for i in range(4)) + 2 * sum(gmunu(eta)[i][j] * rho_partial(eta, x, y, z)[i] * ubar_partial(k_x, k_y, k_z, x, y, z, eta)[i] for i, j in product(range(4), repeat=2)))
    return g

def gbar_i(eta, x, y, z, k_x, k_y, k_z):
    g = 1j * (sum(gmunu(eta)[i][i] * rho_partial(eta, x, y, z)[i] * u(k_x, k_y, k_z, x, y, z, eta) for i in range(4)) + 2 * sum(gmunu(eta)[i][j] * rho_partial(eta, x, y, z)[i] * u_partial(k_x, k_y, k_z, x, y, z, eta)[i] for i, j in product(range(4), repeat=2)))
    return g

def c_n():
    F, G = F_G_N()

    n_vectors = np.array(
            list(product(range(N+1), repeat=3)),
            dtype=int
        )

    momenta = (2.0 * np.pi / L) * n_vectors


    G = np.array(
        len(momenta),
        dtype=np.complex128
    )

    GBar = np.empty_like(G)

    G = [partial(g_i, k_x = k_vec[0], k_y = k_vec[1], k_z = k_vec[2]) for k_vec in momenta]
    GBar = [partial(gbar_i, k_x = k_vec[0], k_y = k_vec[1], k_z = k_vec[2]) for k_vec in momenta]

    c = - sum((F[i][j] * (mu_sigma(partial(causal_propagator_min, o = GBar[i]), partial(causal_propagator_min, o = G[j])) + 1j/2 * symplectic_min(GBar[i], G[j])) 
                + 1/2 * G[i][j] * (mu_sigma(partial(causal_propagator_min, o = G[i]), partial(causal_propagator_min, o = G[j])) + 1j/2 * symplectic_min(G[i], G[j]))
                + 1/2 * G[i][j] * (mu_sigma(partial(causal_propagator_min, o = GBar[i]), partial(causal_propagator_min, o = GBar[j])) + 1j/2 * symplectic_min(GBar[i], GBar[j]))
               )for i, j in np.array(([0, 1, 2, 3], [0, 1, 2, 3]), dtype=int))

    return c


def f_j_minus(k_x, k_y, k_z, x, y, z, eta):
    f_j = - lambda_coupling * 1/a(eta) * greens_operator_min(eta, x, y, z, partial(g_i, k_x = k_x, k_y = k_y, k_z = k_z)) * rho_coupling(x, y, z, eta)
    return f_j

def h_j_minus(k_x, k_y, k_z, x, y, z, eta):
    h_j = g_i(eta, x, y, z, k_x, k_y, k_z)
    return h_j

def f_tilde_j_minus(k_x, k_y, k_z, x, y, z, eta):
    f_tilde_j = - lambda_coupling * 1/a(eta) * greens_operator_min(eta, x, y, z, partial(gbar_i, k_x = k_x, k_y = k_y, k_z = k_z)) * rho_coupling(x, y, z, eta)
    return f_tilde_j

def h_tilde_j_minus(k_x, k_y, k_z, x, y, z, eta):
    h_tilde_j = gbar_i(eta, x, y, z, k_x, k_y, k_z)
    return h_tilde_j


def N_ij(k_x, k_y, k_z, q_x, q_y, q_z):
    N_ij = (mu_omega(partial(causal_propagator_non_min, o = partial(f_tilde_j_minus, k_x = k_x, k_y = k_y, k_z = k_z)), partial(causal_propagator_non_min, o = partial(f_tilde_j_minus, k_x = q_x, k_y = q_y, k_z = q_z)))
    + mu_sigma(partial(causal_propagator_min, o = partial(h_tilde_j_minus, k_x = k_x, k_y = k_y, k_z = k_z)), partial(causal_propagator_min, o = partial(h_j_minus, k_x = k_x, k_y = k_y, k_z = k_z))) 
    + 1j/2 * symplectic_non_min(partial(f_tilde_j_minus, k_x = k_x, k_y = k_y, k_z = k_z), partial(f_j_minus, k_x = q_x, k_y = q_y, k_z = q_z))
    + 1j/2 * symplectic_min(partial(h_tilde_j_minus, k_x = k_x, k_y = k_y, k_z = k_z), partial(h_j_minus, k_x = q_x, k_y = q_y, k_z = q_z)))
    return N_ij

def M_ij(k_x, k_y, k_z, q_x, q_y, q_z):
    M_ij = (mu_omega(partial(causal_propagator_non_min, o = partial(f_j_minus, k_x = k_x, k_y = k_y, k_z = k_z)), partial(causal_propagator_non_min, o = partial(f_j_minus, k_x = q_x, k_y = q_y, k_z = q_z)))
        + mu_sigma(partial(causal_propagator_min, o = partial(h_j_minus, k_x = k_x, k_y = k_y, k_z = k_z)), partial(causal_propagator_min, o = partial(h_j_minus, k_x = k_x, k_y = k_y, k_z = k_z))) 
        + 1j/2 * symplectic_non_min(partial(f_j_minus, k_x = k_x, k_y = k_y, k_z = k_z), partial(f_j_minus, k_x = q_x, k_y = q_y, k_z = q_z))
        + 1j/2 * symplectic_min(partial(h_j_minus, k_x = k_x, k_y = k_y, k_z = k_z), partial(h_j_minus, k_x = q_x, k_y = q_y, k_z = q_z)))
    return M_ij

def N_M():
    n_vectors = np.array(
                list(product(range(N+1), repeat=3)),
                dtype=int
            )
        
    momenta = (2.0 * np.pi / L) * n_vectors


    Nn = np.empty(
        (len(momenta), len(momenta)),
        dtype=np.complex128
    )

    M = np.empty_like(Nn)

    for i, k_vec in momenta:
        for j, q_vec in momenta:
            Nn[i][j] = N_ij(k_vec[0], k_vec[1], k_vec[2], q_vec[0], q_vec[1], q_vec[2])
            M[i][j] = M_ij(k_vec[0], k_vec[1], k_vec[2], q_vec[0], q_vec[1], q_vec[2])

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

def helping_function(t):
    res = 0.0
    if t > 0:
        res = np.exp(- 1/t)
    return res

def charac(epsilon, delta, s):
    erg = helping_function(s - epsilon)/(helping_function(s - epsilon) + helping_function(epsilon + delta - s))
    return erg

def charac_prime(epsilon, delta, s):
    proizlaziti = (1/(s - epsilon)**2 * charac(epsilon, delta, s) - charac(epsilon, delta, s) /(helping_function(s - epsilon) + helping_function(epsilon + delta - s)) 
             * (1/(s + epsilon)**2 * helping_function(s - epsilon) - 1/(epsilon + delta - s) * helping_function(epsilon + delta - s)))
    return proizlaziti

def F_delta_integrand(epsilon, delta, s, t):
    risultato = charac_prime(epsilon, delta, s) * np.exp(- 1j * t * s)
    return risultato

def F_delta(t, delta, epsilon):
    value, err = quad(partial(F_delta_integrand, epsilon = epsilon, delta = delta, t = t), 0, epsilon + delta)
    return value

def probability_integrand(epsilon, delta, t):
    identt = np.eye(2*(N + 1)**3, dtype=np.float64)
    jieguo = 1/t * ((F_delta(t, delta, epsilon) * np.exp(1j * t * c_n()) 
            * (np.linalg.det((identt + S_N(t))/2) * np.linalg.det(identt + 2 * 1j * V_N() * C_N(t))))**(- 1/2)).imag
    return jieguo

def probability(epsilon, delta):
    integral, err = quad(probability_integrand, 0, np.inf, args=(epsilon, delta))
    kekka = 1/2 + 1/np.pi * integral
    return kekka
