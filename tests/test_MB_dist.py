import numpy as np
import pytest  # type: ignore[import-untyped]
from scipy.stats import kstest  # type: ignore[import-untyped]
from typing import List, Tuple, cast
from numpy.typing import NDArray
from numpy.random import Generator, default_rng

from src.MB_dist import MDSimulation, Particle, Species, get_speeds


cutoff: float = 0.05

# Initialize the random number generator
rng: Generator = np.random.default_rng()


def is_maxwell_boltzmann(
    speeds: NDArray[np.float64], masses: NDArray[np.float64], T: float
) -> bool:
    kb: float = 1.38e-23

    def cdf(v: float) -> float:  # Make sure this is float, not bool
        total: float = 0.0
        for m in masses:
            term: flocat = (
                (2 / np.sqrt(np.pi))
                * (np.sqrt(m / (2 * kb * T)) ** 3)
                * v**2
                * np.exp(-m * v**2 / (2 * kb * T))
        )
        total += term
        return total

    d, p_value = kstest(speeds, cdf)
    return (p_value < cutoff)


def setup_simulation() -> Tuple[MDSimulation, float, NDArray[np.float64], float]:
    num_A: int = 500
    num_B: int = 500
    num_C: int = 500
    reaction_probability: float = 1.0
    species_A: Species = Species(name="A", mass=1.0, radius=0.01, color="red")
    species_B: Species = Species(name="B", mass=2.0, radius=0.02, color="blue")
    species_C: Species = Species(name="C", mass=3.0, radius=0.03, color="purple")

    pos_A: NDArray[np.float64] = rng.random((num_A, 2))  # Random positions
    vel_A: NDArray[np.float64] = rng.random((num_A, 2)) - 0.5  # Random velocities

    pos_B: NDArray[np.float64] = rng.random((num_B, 2))  # Random positions
    vel_B: NDArray[np.float64] = rng.random((num_B, 2)) - 0.5  # Random velocities

    pos_C: NDArray[np.float64] = rng.random((num_C, 2))  # Random positions
    vel_C: NDArray[np.float64] = rng.random((num_C, 2)) - 0.5  # Random velocities

    # Create Particle instances with proper type casting
    particles: List[Particle] = (
        [Particle(species_A, cast(NDArray[np.float64], p), cast(NDArray[np.float64], v)) for p, v in zip(pos_A, vel_A)]
        + [Particle(species_B, cast(NDArray[np.float64], p), cast(NDArray[np.float64], v)) for p, v in zip(pos_B, vel_B)]
        + [Particle(species_C, cast(NDArray[np.float64], p), cast(NDArray[np.float64], v)) for p, v in zip(pos_C, vel_C)]
    )

    sim: MDSimulation = MDSimulation(particles, reaction_probability)
    dt: float = 1 / 30  # Time step
    T: float = 300.0  # Temperature

    masses: NDArray[np.float64] = np.array([p.mass for p in particles], dtype=np.float64)

    return (
        sim,
        dt,
        masses,
        T,
    )  # Return the simulation, time step, masses, and temperature


def test_MB() -> None:
    sim, dt, masses, T = setup_simulation()

    # Run the simulation for a sufficient number of steps to reach equilibrium
    num_steps: int = 1000
    for _ in range(num_steps):
        sim.advance(dt)

    # Extract the final speeds from the simulation
    final_speeds: NDArray[np.float64] = get_speeds(sim.particles)

    # Check if the final speeds distribution follows the Maxwell-Boltzmann distribution
    assert is_maxwell_boltzmann(
        final_speeds, masses, T
    ), "Final speed distribution does not match Maxwell-Boltzmann distribution"


def test_basic() -> None:
    assert True


if __name__ == "__main__":
    pytest.main()
