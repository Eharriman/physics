import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.constants import hbar, m_e, g
import os

# Global parameters and config

save_dir = "wave_snapshots"
os.makedirs(save_dir, exist_ok=True)

# Dummy variables for scaling
h_d = 0.5
g_d = 0.5
m_d = 0.5

# Define spatial and temporal spectrum
z_start, z_end = -10, 5
z_nums = 1000
z = np.linspace(z_start, z_end, z_nums)

t_start, t_end = 0, 5
t_nums = 200
t = np.linspace(t_start, t_end, t_nums)

# Initial conditions
z_0 = 0
p_0 = 0
sigma_0 = 1

# Precompute classical trajectory
mean_position = -(z_0 + 0.5 * g * t ** 2)

# Array for mean position in trailing trajectory animation
mean_positions = []

# Save indices for images
save_interval = 0.5
save_times = np.arange(0, t_end + save_interval, save_interval)
save_indices = [np.argmin(np.abs(t - ts)) for ts in save_times]

# The LAWS OF PHYSICS!

def gaussian_wave_packet(z, t, z_0, sigma_0):
    # Calculates a Gaussian wave packet using the simplified physical constants.
    # If you want to use the actual physical constants use h_d -> hbar, m_d -> m_e, g_d -> g
    w = np.sqrt(1 / (1 + ((2 * h_d * t) / m_d) ** 2))
    return np.sqrt(2 / np.pi) * w * np.exp(-2 * (w ** 2) * (z ** 2))

def inertial_wave_function(z, t, z_0, sigma_0):
    # Calculates the wave function for the inertial coordinate system.
    xi = z + 0.5 * g * t ** 2
    psi0 = gaussian_wave_packet(xi, t, z_0, sigma_0)
    phase_factor = np.exp(-1j * m_d * g_d * t * (z + (1 / 6) * g * t ** 2) / hbar)
    return psi0 * phase_factor

def noninertial_wave_function(z_prime, t, z_0, sigma_0):
    # Computes the wave function for the freely falling coordinate system
    psi0 = gaussian_wave_packet(z_prime, t, z_0, sigma_0)
    phase_factor = np.exp(-1j * m_d * g_d * t * (z_prime - (1 / 3) * g * t ** 2) / hbar)
    #return gaussian_wave_packet(z_prime, t, z_0, sigma_0)
    return psi0 * phase_factor

# Use np.abs() for probability distributions.
inertial_prob_dist = np.abs(inertial_wave_function(z[:, np.newaxis], t, z_0, sigma_0)) ** 2
noninertial_prob_dist = np.abs(noninertial_wave_function(z[:, np.newaxis], t, z_0, sigma_0)) ** 2

y_max_inertial = 1
y_max_noninertial = 1

# Animations
def animate_inertial_vs_noninertial():
    fig, (x1, x2) = plt.subplots(1, 2, figsize=(12, 6))

    def update(frame):
        x1.clear()
        x2.clear()

        # Compute mean position
        mean_pos = -0.5 * g * t[frame] ** 2
        mean_positions.append(mean_pos)

        if len(mean_positions) > 20:
            mean_positions.pop(0)

        # Inertial frame
        x1.plot(z, inertial_prob_dist[:, frame], label=f"t = {t[frame]:.2f} s")
        x1.set_title("Inertial Frame")
        x1.set_xlabel("z (m)")
        x1.set_ylabel("|ψ(z, t)|²")
        x1.set_ylim(0, y_max_inertial)

        # Fading
        alpha_values = np.linspace(0.1, 1, len(mean_positions))
        for i in range(len(mean_positions) - 1):
            x1.plot([mean_positions[i], mean_positions[i+1]],
                    [y_max_inertial * 0.7, y_max_inertial * 0.7], 'r-', alpha=alpha_values[i])

        x1.plot([mean_pos, mean_pos], [0, y_max_inertial * 0.8], 'r--', alpha=0.8)
        x1.legend()

        # Non-inertial frame
        x2.plot(z, noninertial_prob_dist[:, frame], label=f"t = {t[frame]:.2f} s")
        x2.set_title("Non-Inertial Frame")
        x2.set_xlabel("z' (m)")
        x2.set_ylabel("|ψ(z', t)|²")
        x2.set_ylim(0, y_max_noninertial)
        x2.legend()

        plt.tight_layout()

    ani = FuncAnimation(fig, update, frames=t_nums, interval=50, blit=False)
    plt.show()

def animate_wavefunction_3D():
    # Work in progress for 3D animal. First time using mplot3d
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        ax.clear()
        ax.plot3D(z, [t[frame]] * len(z), inertial_prob_dist[:, frame], color='b')
        ax.set_xlabel("Position (z)")
        ax.set_ylabel("Time (t)")
        ax.set_zlabel("|ψ(z, t)|²")
        ax.set_title("3D Wavefunction Evolution")

    ani = FuncAnimation(fig, update, frames=t_nums, interval=50, blit=False)
    plt.show()

def animate_particle_vs_wave():
    # Side-by-side of particle and wave packet falling
    fig, ax = plt.subplots(figsize=(8, 6))

    def update(frame):
        ax.clear()
        ax.plot(z, inertial_prob_dist[:, frame], 'b-', label="Wavefunction Probability")
        particle_x = mean_position[frame]
        ax.plot([particle_x], [y_max_inertial * 0.8], 'ro', label="Classical Particle")
        ax.set_title("Classical Particle vs. Quantum Wave")
        ax.set_xlabel("z (m)")
        ax.set_ylabel("|ψ(z, t)|²")
        ax.legend()

    ani = FuncAnimation(fig, update, frames=t_nums, interval=50, blit=False)
    plt.show()


def simulate_measurements(trial_list):
    # Function simulates position measurements. n trials increases and the histogram smooths out to recover
    # the pdf. This is using the inertial pdf at t = 0
    if isinstance(trial_list, int):  # Allow single integer input
        trial_list = [trial_list]

    num_subplots = len(trial_list)
    fig, axes = plt.subplots(1, num_subplots, figsize=(5 * num_subplots, 4))  # Remove sharey=True

    if num_subplots == 1:
        axes = [axes]  # Ensure it's always iterable

    pdf = inertial_prob_dist[:, 0]
    pdf /= np.trapezoid(pdf, z)  # Normalize

    for i, n in enumerate(trial_list):
        # Select random measurements bounded by wave packet
        samples = np.random.choice(z, size=n, p=pdf / np.sum(pdf))

        # Histogram plots for trials
        counts, bins, _ = axes[i].hist(samples, bins=40, alpha=0.6, color='b', label=f"n={n}")

        # Scale y-axis for each histogram by trial size
        axes[i].set_ylim(0, max(counts) * 1.1)
        axes[i].set_xlabel("Position (z)")
        axes[i].axvline(0, color='k', linestyle='--', alpha=0.7)
        axes[i].legend()

    axes[0].set_ylabel("Number of Measurements")
    plt.suptitle("Position Measurements over Identical Trials (n = 10, 100, 1000, 100000")
    plt.show()


# Animation method. *n_trials_list is an optional parameter

def run_animation(animation_type="all", *n_trials_list):
    if animation_type == "inertial_vs_noninertial":
        animate_inertial_vs_noninertial()
    elif animation_type == "wavefunction_3D":
        animate_wavefunction_3D()
    elif animation_type == "particle_vs_wave":
        animate_particle_vs_wave()
    elif animation_type == "simulate_measurements":
            simulate_measurements(*n_trials_list)
    elif animation_type == "all":
        animate_inertial_vs_noninertial()
        animate_wavefunction_3D()
        animate_particle_vs_wave()
    else:
        print(f"Unknown animation type: {animation_type}")

n_trials_list = [10, 100, 1000]

# Uncomment the animation you want to watch here
# run_animation("inertial_vs_noninertial")
# run_animation("wavefunction_3D")
#run_animation("particle_vs_wave")
# run_animation("simulate_measurements",n_trials_list)
# run_animation("all")

# Run the simulation with increasing number of measurements
#n_trials_list = [10, 100, 1000, 100000]
# Example usage:
#simulate_measurements([10, 100, 1000, 100000])
