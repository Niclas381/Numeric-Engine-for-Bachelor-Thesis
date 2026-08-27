import numpy as np
from scipy.integrate import simpson
from mode_solver_native import load_mode      #take new mode solver from riccaticpp for second order ODE's of similar form
import h5py
from itertools import product
from scipy.linalg import expm, solve

#parameters of physical model

L = 1  #to be chosen later (defines spatial volume)

m = 1  #mass of probe, chose later

N = 1 #number of k in each direction

with h5py.File("mode_solutions_native.h5", "r") as h5:
        obj = h5["runs/quadratic_test_002/background/eta"]

        if not isinstance(obj, h5py.Dataset):
            raise TypeError("The eta path does not contain an HDF5 dataset")

        eta_values = np.asarray(obj[:], dtype=np.float64)

x = np.linspace(1, L, 100)
y = np.linspace(1, L, 100)
z = np.linspace(1, L, 100)

X, Y, Z = np.meshgrid(x, y, z, indexing="ij", sparse=True)

#scale factor
def a(eta):
    return eta**2        

def a_prime(eta):
    return 2 * eta  

def a_dprime(eta):
    return 2

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
    u_0 = 1
    u_1 = 1
    u_2 = 1
    u_3 = 1
    return np.array([
        u_0, 
        u_1,
        u_2,
        u_3], dtype= np.float64)

def chi(k, target_eta):                                     #Note to myself: target_eta has to be in the array eta, so check the grid with the actual mode solver
    k_found, eta, chi_k, chi_prime_k = load_mode(
        filename = "mode_solutions_native.h5",
        run_name = "test",
        k = k
    )

    i = np.searchsorted(eta, target_eta)
    return chi_k[i]

def chi_prime(k, target_eta):
    k_found, eta, chi_k, chi_prime_k = load_mode(
            filename = "mode_solutions_native.h5",
            run_name = "test",
            k = k
        )
    
    i = np.searchsorted(eta, target_eta)
    return chi_prime_k[i]


#frequencies of the field
def u(k_x, k_y, k_z, x, y, z, eta):
    u = 1/np.sqrt(L**3) * chi(np.sqrt(k_x**2 + k_y**2 + k_z**2), eta) / a(eta) * np.exp(1j * (k_x * x + k_y * y + k_z * z))
    return u

def u_partial_x(k_x, k_y, k_z, x, y, z, eta):
    u = 1j * k_x * 1/np.sqrt(L**3) * chi(np.sqrt(k_x**2 + k_y**2 + k_z**2), eta) / a(eta) * np.exp(1j * (k_x * x + k_y * y + k_z * z))
    return u

def u_partial_y(k_x, k_y, k_z, x, y, z, eta):
    u = 1j * k_y * 1/np.sqrt(L**3) * chi(np.sqrt(k_x**2 + k_y**2 + k_z**2), eta) / a(eta) * np.exp(1j * (k_x * x + k_y * y + k_z * z))
    return u

def u_partial_z(k_x, k_y, k_z, x, y, z, eta):
    u = 1j * k_z * 1/np.sqrt(L**3) * chi(np.sqrt(k_x**2 + k_y**2 + k_z**2), eta) / a(eta) * np.exp(1j * (k_x * x + k_y * y + k_z * z))
    return u

def u_partial_eta(k_x, k_y, k_z, x, y, z, eta):
    u = 1/np.sqrt(L**3) * (chi_prime(np.sqrt(k_x**2 + k_y**2 + k_z**2), eta) / a(eta) - chi(np.sqrt(k_x**2 + k_y**2 + k_z**2), eta) * a_prime(eta)/(a(eta)**2) ) * np.exp(1j * (k_x * x + k_y * y + k_z * z))
    return u

def ubar(k_x, k_y, k_z, x, y, z, eta):       #lol
    u = 1/np.sqrt(L**3) * (chi(np.sqrt(k_x**2 + k_y**2 + k_z**2), eta).real - 1j * chi(np.sqrt(k_x**2 + k_y**2 + k_z**2), eta).imag)  / a(eta) * np.exp(1j * (k_x * x + k_y * y + k_z * z))
    return u

def ubar_partial_x(k_x, k_y, k_z, x, y, z, eta):       #lol
    u = - 1j * k_x * 1/np.sqrt(L**3) * (chi(np.sqrt(k_x**2 + k_y**2 + k_z**2), eta).real - 1j * chi(np.sqrt(k_x**2 + k_y**2 + k_z**2), eta).imag)  / a(eta) * np.exp(- 1j * (k_x * x + k_y * y + k_z * z))
    return u

def ubar_partial_y(k_x, k_y, k_z, x, y, z, eta):       #lol
    u = - 1j * k_y * 1/np.sqrt(L**3) * (chi(np.sqrt(k_x**2 + k_y**2 + k_z**2), eta).real - 1j * chi(np.sqrt(k_x**2 + k_y**2 + k_z**2), eta).imag)  / a(eta) * np.exp(- 1j * (k_x * x + k_y * y + k_z * z))
    return u

def ubar_partial_z(k_x, k_y, k_z, x, y, z, eta):       #lol
    u = - 1j * k_z * 1/np.sqrt(L**3) * (chi(np.sqrt(k_x**2 + k_y**2 + k_z**2), eta).real - 1j * chi(np.sqrt(k_x**2 + k_y**2 + k_z**2), eta).imag)  / a(eta) * np.exp(- 1j * (k_x * x + k_y * y + k_z * z))
    return u

def ubar_partial_eta(k_x, k_y, k_z, x, y, z, eta):
    u = 1/np.sqrt(L**3) * ((chi_prime(np.sqrt(k_x**2 + k_y**2 + k_z**2), eta).real - 1j * chi_prime(np.sqrt(k_x**2 + k_y**2 + k_z**2), eta).imag)/ a(eta) - (chi(np.sqrt(k_x**2 + k_y**2 + k_z**2), eta).real - 1j * chi(np.sqrt(k_x**2 + k_y**2 + k_z**2), eta)) * a_prime(eta)/(a(eta)**2) ) * np.exp(1j * (k_x * x + k_y * y + k_z * z))
    return u

#compact smearing function
def f(eta, x, y, z):
    return eta

def integrand_F_N(k1_x, k1_y, k1_z, k2_x, k2_y, k2_z, eta_n, X, Y, Z):
        #smearing function
        f_smearing = f(eta_n, X, Y, Z)

        #velocity
        vel = umu(eta_n)

        #modes
        ubar1 = ubar(k1_x, k1_y, k1_z, X, Y, Z, eta_n)
        u2 = u(k2_x, k2_y, k2_z, X, Y, Z, eta_n)
        ubar1_partial = np.array([
                    ubar_partial_x(k1_x, k1_y, k1_z, X, Y, Z, eta_n),
                    ubar_partial_y(k1_x, k1_y, k1_z, X, Y, Z, eta_n),
                    ubar_partial_z(k1_x, k1_y, k1_z, X, Y, Z, eta_n),
                    ubar_partial_eta(k1_x, k1_y, k1_z, X, Y, Z, eta_n)], dtype=np.complex128)
        u2_partial = np.array([
            u_partial_x(k2_x, k2_y, k2_z, X, Y, Z, eta_n),
            u_partial_y(k2_x, k2_y, k2_z, X, Y, Z, eta_n),
            u_partial_z(k2_x, k2_y, k2_z, X, Y, Z, eta_n),
            u_partial_eta(k2_x, k2_y, k2_z, X, Y, Z, eta_n)], dtype=np.complex128)

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

        return f_smearing * (a * b - 1/2 * e * (c + m**2 * d))

def integrand_G_N(k1_x, k1_y, k1_z, k2_x, k2_y, k2_z, eta_n, X, Y, Z):
        #smearing function
        f_smearing = f(eta_n, X, Y, Z)

        #velocity
        vel = umu(eta_n)

        #modes
        ubar1 = ubar(k1_x, k1_y, k1_z, X, Y, Z, eta_n)
        u2 = u(k2_x, k2_y, k2_z, X, Y, Z, eta_n)
        ubar1_partial = np.array([
                    u_partial_x(k1_x, k1_y, k1_z, X, Y, Z, eta_n),
                    u_partial_y(k1_x, k1_y, k1_z, X, Y, Z, eta_n),
                    u_partial_z(k1_x, k1_y, k1_z, X, Y, Z, eta_n),
                    u_partial_eta(k1_x, k1_y, k1_z, X, Y, Z, eta_n)], dtype=np.complex128)
        u2_partial = np.array([
            u_partial_x(k2_x, k2_y, k2_z, X, Y, Z, eta_n),
            u_partial_y(k2_x, k2_y, k2_z, X, Y, Z, eta_n),
            u_partial_z(k2_x, k2_y, k2_z, X, Y, Z, eta_n),
            u_partial_eta(k2_x, k2_y, k2_z, X, Y, Z, eta_n)], dtype=np.complex128)

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

        return f_smearing * (a * b - 1/2 * e * (c + m**2 * d))

def integrate_spatial(values, x, y, z):
    result = simpson(values, x = z, axis=2)
    result = simpson(result, x = y, axis=1)
    result = simpson(result, x = x, axis=0)

#helping functions for the energy density
def F_N_element(k_x, k_y, k_z, q_x, q_y, q_z):
    spatial_integrals = np.empty(len(eta_values), dtype=np.complex128)

    for n, eta_n in enumerate(eta_values):
        values = integrand_F_N(k_x, k_y, k_z, q_x, q_y, q_z, eta_n, X, Y, Z)

        spatial_integrals[n] = integrate_spatial(values, x, y, z)

    I = simpson(spatial_integrals, x = eta_values)

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

def c_n():
    pass

def sigma_prime():
    pass

def charac(epsilon, delta):
    pass

def F_delta(t, delta, epsilon):
    pass

def probability(epsilon):
    pass
