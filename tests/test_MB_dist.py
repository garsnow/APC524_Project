import pytest
import numpy as np
from md_simulation import MDSimulation, get_speeds

def is_maxwell_boltzmann(speeds, m, T):
    from scipy.stats import kstest
    kb = 1.38e-23
    a = np.sqrt(m / (2 * kb * T))
    cdf = lambda v: (2 / np.sqrt(np.pi)) * (a ** 3) * v ** 2 * np.exp(-a ** 2 * v ** 2)
    d, p_value = kstest(speeds, cdf)
    return p_value < 0.05

@pytest.fixture
def setup_simulation():
    n = 1000
    rscale = 5.e6
    r = 2e-10 * rscale
    tscale = 1e9
    sbar = 353 * rscale / tscale
    dt = 1/30
    m = 1
    pos = np.random.random((n, 2))
    theta = np.random.random(n) * 2 * np.pi
    s0 = sbar * np.random.random(n)
    vel = (s0 * np.array((np.cos(theta), np.sin(theta)))).T
    sim = MDSimulation(pos, vel, r, m)
    return sim, dt, m, 300  # Return the simulation, time step, mass, and temperature

def test_maxwell_boltzmann_distribution(setup_simulation):
    sim, dt, m, T = setup_simulation
    for _ in range(1000):  # Run the simulation for 1000 steps
        sim.advance(dt)
    speeds = get_speeds(sim.vel)
    assert is_maxwell_boltzmann(speeds, m, T), "Speed distribution does not match Maxwell-Boltzmann distribution"

if __name__ == "__main__":
    pytest.main()

