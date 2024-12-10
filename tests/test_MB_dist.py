import numpy as np
import pytest
from scipy.stats import kstest

from MD_dist import MDSimulation, get_speeds, particle_simulator

cutoff = 0.05


def is_maxwell_boltzmann(speeds: np.ndarray, m: float, T: float) -> bool:
    """
    Checks if the speed distribution from the MB_dist simulation is sampled from a Maxwell_Boltzmann distribution. 

    Args:
        speeds (array): speeds of particles.
        m (float): particle mass (assumed identical between particles).
        T (float): temperature.

    Returns:
        bool
    """
    kb = 1.38e-23
    a = np.sqrt(m / (2 * kb * T))
    cdf = lambda v: (2 / np.sqrt(np.pi)) * (a**3) * v**2 * np.exp(-a**2 * v**2)
    d, p_value = kstest(speeds, cdf)
    return p_value < cutoff


@pytest.fixture
def test_maxwell_boltzmann_distribution() -> None:
    """
    Checks if the speed distribution from the MB_dist simulation is sampled from a Maxwell_Boltzmann distribution. 

    Args:
        setup_simulation

    Returns:
        None
    """
    number_of_particles = 1000
    time_step = 30
    particle_mass = 1
    sim = particle_simulator(number_of_particles, time_step, particle_mass)
    speeds = get_speeds(sim.vel)
    assert is_maxwell_boltzmann(speeds, particle_mass, T=300), "Does not match Maxwell-Boltzmann distribution"


if __name__ == "__main__":
    pytest.main()
