# Required modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.constants import hbar, m_e, g

# Dummy variables to make the standard deviation bigger
# 10 was pretty good but dropped pretty quickly
# 5 is about the same, try 1 again. Trying 0.5
h_d = 0.5
g_d = 0.5
m_d = 0.5

# Define spatial spectrum (variable depending on curve width)
#z_start, z_end = -50, 50
z_start, z_end = -100, 15
z_nums = 1000 # Number of points on range
z = np.linspace(z_start, z_end, z_nums)

# Define temporal spectrum
t_start, t_end = 0, 10
t_nums = 100
t = np.linspace(t_start, t_end, t_nums)

# Initial conditions
z_0 = 0
p_0 = 0
sigma_0 = 1

def gaussian_wave_packet(z, t, z_0, sigma_0):
    # Define the standard deviation
    #sigma_t = 1/np.sqrt(1 / (1 + ((2 * hbar * t)/m_e) ** 2))
    sigma_t = 1 / np.sqrt(1 / (1 + ((2 * h_d * t) / m_d) ** 2)) #Using dummy variables
    #w = np.sqrt(1 / (1 + ((2 * hbar * t)/m_e) ** 2))
    w = np.sqrt(1 / (1 + ((2 * h_d * t) / m_d) ** 2)) #Using dummy variables

    phi_0 = np.sqrt(2 / np.pi) * w * np.exp(-2 * (w ** 2) * (z ** 2))

    return phi_0

def inertial_wave_function(z, t, z_0, sigma_0):
    # Transformed coordinate
    xi = z + 0.5 * g * t ** 2

    # Free Gaussian wave packet
    psi0 = gaussian_wave_packet(xi, t, z_0, sigma_0)

    # Gravitational phase factor
    #phase_factor = np.exp(-1j * m_e * g * t * (z + (1 / 6) * g * t ** 2) / hbar)
    phase_factor = np.exp(-1j * m_d * g_d * t * (z + (1 / 6) * g * t ** 2) / hbar)

    # Full wave function
    psi = psi0 * phase_factor
    return psi

def noninertial_wave_function(z_prime, t, z_0, sigma_0):
    psi = gaussian_wave_packet(z_prime, t, z_0, sigma_0)
    return psi

# Compute the probability distributions
inertial_prob_dist = np.abs(inertial_wave_function(z[:, np.newaxis], t, z_0, sigma_0)) ** 2
noninertial_prob_dist = np.abs(noninertial_wave_function(z[:, np.newaxis], t, z_0, sigma_0)) ** 2

# Plot results
fig, (x1, x2) = plt.subplots(1, 2, figsize=(12, 6))

# Frame update
def update(frame):
    x1.clear()
    x2.clear()

    # Intertial frame (on the left)
    x1.plot(z, inertial_prob_dist[:, frame], label=f"t = {t[frame]:.2f} s")
    x1.set_title("Inertial Frame")
    x1.set_xlabel("z (m)")
    x1.set_ylabel("|ψ(z, t)|²")
    #x1.set_ylim(0, 0.5)
    x1.set_ylim(0, 0.5)
    x1.legend()

    # Non-inertial frame
    x2.plot(z, noninertial_prob_dist[:, frame], label=f"t = {t[frame]:.2f} s")
    x2.set_title("Non-Inertial Frame")
    x2.set_xlabel("z' (m)")
    x2.set_ylabel("|ψ(z', t)|²")
    #x2.set_ylim(0, 0.5)
    x2.set_ylim(0, 0.5)
    x2.legend()

    plt.tight_layout()

# FuncAnimation to update the frame of the matplot
ani = FuncAnimation(fig, update, frames=t_nums, interval=50, blit=False)

# Display the animated plot
plt.show()



