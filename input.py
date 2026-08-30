#import modules
from scipy.integrate import nquad
import numpy as np

#needed variables
def box_length_L():
    L = 1
    return L

def lambda_coupl():
    lambda_ = 1
    return lambda_

def mass_probe_and_system():
    m_probe = 1
    m_system = 1
    return m_probe, m_system

def number_of_modes_per_direction():
    number = 1
    return number


#needed functions
def a(eta):
    res = eta**2
    return res

def a_prime(eta):
    res = 2 * eta
    return res

def a_2prime(eta):
    res = 2
    return res

def compact_smearing_function(eta, x, y, z):
    f = 1
    return f

def function_for_compact_coupling_region(eta, x, y, z):
    rho = 1
    return rho

def function_to_obtain_mode(eta, x, y, z):
    rho = 1
    rho_partial = 1
    rho_2partial = 1
    return rho, rho_partial, rho_2partial

def four_velocity(eta):
    u_0 = 1
    u_1 = 1
    u_2 = 1
    u_3 = 1
    return np.array([
        u_0, 
        u_1,
        u_2,
        u_3], dtype= np.float64)

def t_grid(epsilon, delta):
    return np.linspace(0.0, epsilon + delta, 201)

#real part of scalar product provided by state inpute
def mu_sigma(f, g):
    mu_sigma_integral_core = lambda eta, x, y, z: f(eta, x, y, z) * g(eta, x, y, z)
    return nquad(mu_sigma_integral_core, [[- np.inf, np.inf]]*4)[0]
    

def mu_omega(f, g):
    mu_omega_integral_core = lambda eta, x, y, z: f(eta, x, y, z) * g(eta, x, y, z)
    return  nquad(mu_omega_integral_core, [[- np.inf, np.inf]]*4)[0]
    