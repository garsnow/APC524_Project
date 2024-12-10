import os
import sys
import pytest
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.MB_dist import get_speeds, get_KE, MDSimulation, Histogram

# Testing the get_speeds function
@pytest.mark.parametrize(
    "vel, expected_speeds",
    [
            # basic positive vel
            (
                np.array([
                    [3.0, 4.0],
                    [0.0, 0.0],
                    [1.0, 1.0],
                ]),
                np.array([5.0, 0.0, np.sqrt(2)])
            ),
            # mixed neg, pos, zero velocities
            (
                np.array([
                    [-3.0, -4.0],
                    [0.0, -2.0],
                    [3.0, -1.0],
                ]),
                np.array([5.0, 2.0, np.sqrt(10)])
            ),
            # zero velocities
            (
                np.array([
                    [0.0, 0.0]
                ]),
                np.array([0.0])
            ),
            # large and small velocities
            (
                np.array([
                    [1e6, 1e6],
                    [1e-10, 1e-10],
                ]),
                np.array([np.sqrt(2)*1e6, np.sqrt(2)*1e-10])
            ),
    ]
)
def test_get_speeds(vel, expected_speeds):
    computed_speeds = get_speeds(vel)
    np.testing.assert_array_almost_equal(computed_speeds, expected_speeds)

# Test with an empty array
def test_get_speeds_empty():
    vel = np.array([]).reshape(0, 2)
    expected_speeds = np.array([])
                    
    computed_speeds = get_speeds(vel)
    np.testing.assert_array_equal(computed_speeds, expected_speeds)

# Testing the get_KE function
def test_get_KE():
    m = 2.0
    speeds = np.array([3.0, 4.0])
    expected_ke = 0.5 * m * (3.0**2 + 4.0**2)
    computed_ke = get_KE(m, speeds)
    assert computed_ke == expected_ke

# Testing MDSimulation Class
def test_MDSImulation_init():
    pos = np.array([
        [0.1, 0.1],
        [0.9, 0.9]
    ])
    vel = np.array([
        [1.0, 0.0],
        [0.0, -1.0]
    ])
    r = 0.05 
    m = 1.0

    sim = MDSimulation(pos, vel, r, m)

    assert np.array_equal(sim.pos, pos)
    assert np.array_equal(sim.vel, vel)
    assert sim.n == 2
    assert sim.r == r
    assert sim.m == m
    assert sim.nsteps == 0

# Testing MDsimulation.advance
# with collision
def test_MDSimulation_advance():
    pos = np.array([
        [0.4, 0.5],
        [0.6, 0.5]
    ])
    vel = np.array([
        [1.0, 0.0],
        [-1.0, 0.0]
    ])
    r = 0.05
    m = 1.0

    sim = MDSimulation(pos, vel, r, m)
    dt = 0.1

    sim.advance(dt)
    expected_pos = pos + vel * dt

    expected_vel = np.array([
        [-1.0, 0.0],
        [1.0, 0.0]
    ])

    np.testing.assert_array_almost_equal(sim.pos, expected_pos)
    np.testing.assert_array_almost_equal(sim.vel, expected_vel)
    assert sim.nsteps == 1

# without collision
def test_MDSimulation_advance():
    pos = np.array([
        [0.1, 0.1],
        [0.9, 0.9]
    ])
    vel = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ])
    r = 0.05
    m = 1.0

    sim = MDSimulation(pos, vel, r, m)
    dt = 0.1

    sim.advance(dt)
    expected_pos = pos + vel * dt
    expected_vel = vel.copy()

    np.testing.assert_array_almost_equal(sim.pos, expected_pos)
    np.testing.assert_array_almost_equal(sim.vel, expected_vel)
    assert sim.nsteps == 1

#collision with wall (MDSimulation boundary reflection)
def test_MDSimulation_boundary_reflection():
    pos = np.array([
        [0.05, 0.5],
        [0.95, 0.5]
    ])
    vel = np.array([
        [-1.0, 0.0],
        [1.0, 0.0]
    ])

    r = 0.05
    m = 1.0

    sim = MDSimulation(pos, vel, r, m)
    dt = 0.1

    sim.advance(dt)
    expected_pos = pos + vel * dt
    expected_vel = np.array([
        [1.0, 0.0],
        [-1.0, 0.0]
    ])

    np.testing.assert_array_almost_equal(sim.pos, expected_pos)
    np.testing.assert_array_almost_equal(sim.vel, expected_vel)
    assert sim.nsteps == 1

# Testing Histogram class initilization
def test_Histogram_init():
    data = np.array([1, 2, 2, 3, 3, 3])
    xmax = 4
    nbars = 3
    density = False

    hist_obj = Histogram(data, xmax, nbars, density)

    expected_hist, expected_bins = np.histogram(data, bins=np.linspace(0, 4, 3), density=False)
    assert np.array_equal(hist_obj.hist, expected_hist)
    assert np.array_equal(hist_obj.bins, expected_bins)
    assert hist_obj.nbars == nbars
    assert hist_obj.density == density

# Testing Histogram.update method
def test_Histogram_update():
    data_initial = np.array([1, 2, 2, 3, 3, 3])
    xmax = 4
    nbars = 3
    density = False

    hist_obj = Histogram(data_initial, xmax, nbars, density)
    data_new = np.array([2, 3, 4, 4, 4, 4])
    hist_obj.update(data_new)

    expected_hist, _ = np.histogram(data_new, bins=np.linspace(0, 4, 3), density=False)
    assert np.array_equal(hist_obj.hist, expected_hist)
    expected_top = np.zeros(len(expected_hist)) + expected_hist
    np.testing.assert_array_almost_equal(hist_obj.top, expected_top)
    
