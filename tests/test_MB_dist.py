import pytest
import numpy as np
from scipy.stats import kstest
from src.MB_dist import MDSimulation, get_speeds, Species, Particle

cutoff: float = 0.05


def is_maxwell_boltzmann(speeds: np.ndarray, masses: np.ndarray, T: float) -> bool:
    kb: float = 1.38e-23
    cdf = lambda v: sum(
        [
            (2 / np.sqrt(np.pi))
            * (np.sqrt(m / (2 * kb * T)) ** 3)
            * v**2
            * np.exp(-m * v**2 / (2 * kb * T))
            for m in masses
        ]
    )
    d, p_value = kstest(speeds, cdf)
    return p_value < cutoff


def setup_simulation():
    num_A = 500
    num_B = 500
    num_C = 500
    species_A = Species(name="A", mass=1.0, radius=0.01, color="red")
    species_B = Species(name="B", mass=2.0, radius=0.02, color="blue")
    species_C = Species(name="C", mass=3.0, radius=0.03, color="purple")

    pos_A = np.random.rand(num_A, 2)  # Random positions
    vel_A = np.random.rand(num_A, 2) - 0.5  # Random velocities

    pos_B = np.random.rand(num_B, 2)  # Random positions
    vel_B = np.random.rand(num_B, 2) - 0.5  # Random velocities

    pos_C = np.random.rand(num_C, 2)  # Random positions
    vel_C = np.random.rand(num_C, 2) - 0.5  # Random velocities

    particles = (
        [Particle(species_A, p, v) for p, v in zip(pos_A, vel_A)]
        + [Particle(species_B, p, v) for p, v in zip(pos_B, vel_B)]
        + [Particle(species_C, p, v) for p, v in zip(pos_C, vel_C)]
    )

    sim = MDSimulation(particles)
    dt = 1 / 30  # Time step
    return (
        sim,
        dt,
        np.array([p.mass for p in particles]),
        300,
    )  # Return the simulation, time step, masses, and temperature


def test_MB():
    sim, dt, masses, T = setup_simulation()

    # Run the simulation for a sufficient number of steps to reach equilibrium
    num_steps = 1000
    for step in range(num_steps):
        sim.advance(dt)

    # Extract the final speeds from the simulation
    final_speeds = get_speeds(sim.particles)

    # Check if the final speeds distribution follows the Maxwell-Boltzmann distribution
    assert is_maxwell_boltzmann(
        final_speeds, masses, T
    ), "Final speed distribution does not match Maxwell-Boltzmann distribution"


def test_basic():
    assert True


if __name__ == "__main__":
    pytest.main()
