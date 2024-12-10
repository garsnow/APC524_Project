import numpy as np
import pytest
from scipy.stats import kstest

from src.example.MD_dist import MDSimulation, get_speeds

cutoff = 0.05


def is_maxwell_boltzmann(speeds: np.ndarray, m: float, T: float) -> bool:
    '''
    Checks if the speed distribution from the MB_dist simulation is sampled from a Maxwell_Boltzmann distribution. 

    Args:
        speeds (array): speeds of particles.
        m (float): particle mass (assumed identical between particles).
        T (float): temperature.

    Returns:
        bool
    '''
    kb = 1.38e-23
    a = np.sqrt(m / (2 * kb * T))
    cdf = lambda v: (2 / np.sqrt(np.pi)) * (a**3) * v**2 * np.exp(-a**2 * v**2)
    d, p_value = kstest(speeds, cdf)
    return p_value < cutoff


@pytest.fixture
def setup_simulation() -> tuple:
    '''
    Initializes simulation. 

    Args:
        None

    Returns:
        sim
        dt
        m

    '''
    n = 1000
    rscale = 5.0e6
    r = 2e-10 * rscale
    tscale = 1e9
    sbar = 353 * rscale / tscale
    dt = 1 / 30
    m = 1
    T = 300
    pos = np.random.random((n, 2))
    theta = np.random.random(n) * 2 * np.pi
    s0 = sbar * np.random.random(n)
    vel = (s0 * np.array((np.cos(theta), np.sin(theta)))).T
    sim = MDSimulation(pos, vel, r, m)
    return sim, dt, m, T  # Return the simulation, time step, mass, and temperature


def test_maxwell_boltzmann_distribution(setup_simulation: tuple) -> None:
    '''
    Checks if the speed distribution from the MB_dist simulation is sampled from a Maxwell_Boltzmann distribution. 

    Args:
        setup_simulation

    Returns:
        None
    '''
    sim, dt, m, T = setup_simulation
    for _ in range(1000):  # Run the simulation for 1000 steps
        sim.advance(dt)
    speeds = get_speeds(sim.vel)
    assert is_maxwell_boltzmann(speeds, m, T), "Does not match Maxwell-Boltzmann distribution"


if __name__ == "__main__":
    pytest.main()
