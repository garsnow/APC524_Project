import os
import sys
import pytest
import numpy as np
from scipy.stats import kstest

# Add 'src/' directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from src.MB_dist import get_speeds, get_KE, MDSimulation, Histogram
from src.species_and_particle import Species, Particle

X, Y = 0, 1

@pytest.fixture
def species_A():
    return Species(name="A", mass=1.0, radius=0.5, color="red")

@pytest.fixture
def species_B():
    return Species(name="B", mass=2.0, radius=0.5, color="blue")

@pytest.fixture
def species_C():
    return Species(name="C", mass=3.0, radius=0.5, color='purple')

@pytest.fixture
def particle_A(species_A):
    pos = np.array([0.1, 0.1])
    vel = np.array([1.0, 0.0])
    return Particle(species_A, pos, vel)

@pytest.fixture
def particle_B(species_B):
    pos = np.array([0.9, 0.9])
    vel = np.array([-1.0, 0.0])
    return Particle(species_B, pos, vel)

@pytest.fixture
def particle_C(species_C):
    pos = np.array([0.5, 0.5])
    vel = np.array([0.0, 0.0])
    return Particle(species_C, pos, vel)

@pytest.fixture
def simulation(particle_A, particle_B):
    particles = [particle_A, particle_B]
    return MDSimulation(particles)

# Testing the get_speeds function
@pytest.mark.parametrize(
    "particles, expected_speeds",
    [
        # Basic positive velocities
        (
            [
                Particle(Species("A", 1.0, 0.01, "red"), np.array([0.2, 0.2]), np.array([3.0, 4.0])),
                Particle(Species("B", 2.0, 0.02, "blue"), np.array([0.3, 0.3]), np.array([0.0, 0.0])),
                Particle(Species("A", 1.0, 0.01, "red"), np.array([0.4, 0.4]), np.array([1.0, 1.0])),
            ],
            np.array([5.0, 0.0, np.sqrt(2)])
        ),
        # Mixed negative, positive, zero velocities
        (
            [
                Particle(Species("A", 1.0, 0.01, "red"), np.array([0.2, 0.2]), np.array([-3.0, -4.0])),
                Particle(Species("B", 2.0, 0.02, "blue"), np.array([0.3, 0.3]), np.array([0.0, -2.0])),
                Particle(Species("A", 1.0, 0.01, "red"), np.array([0.4, 0.4]), np.array([3.0, -1.0])),
            ],
            np.array([5.0, 2.0, np.sqrt(10)])
        ),
        # Zero velocities
        (
            [
                Particle(Species("A", 1.0, 0.01, "red"), np.array([0.2, 0.2]), np.array([0.0, 0.0]))
            ],
            np.array([0.0])
        ),
        # Large and small velocities
        (
            [
                Particle(Species("A", 1.0, 0.01, "red"), np.array([0.2, 0.2]), np.array([1e6, 1e6])),
                Particle(Species("B", 2.0, 0.02, "blue"), np.array([0.3, 0.3]), np.array([1e-10, 1e-10])),
            ],
            np.array([np.sqrt(2)*1e6, np.sqrt(2)*1e-10])
        ),
    ]
)
def test_get_speeds(particles, expected_speeds):
    computed_speeds = get_speeds(particles)
    np.testing.assert_array_almost_equal(computed_speeds, expected_speeds)

# Test with an empty list of particles
def test_get_speeds_empty():
    particles = []
    expected_speeds = np.array([])
                    
    computed_speeds = get_speeds(particles)
    np.testing.assert_array_equal(computed_speeds, expected_speeds)

# Testing the get_KE function
def test_get_KE():
    species_A = Species(name="A", mass=1.0, radius=0.01, color="red")
    species_B = Species(name="B", mass=2.0, radius=0.02, color="blue")
    
    particle_A = Particle(species_A, np.array([0.1, 0.1]), np.array([3.0, 4.0]))
    particle_B = Particle(species_B, np.array([0.9, 0.9]), np.array([0.0, 0.0]))
    
    speeds = get_speeds([particle_A, particle_B])
    m = 2.0  # Average mass or specific mass as per your simulation logic
    expected_ke = 0.5 * m * np.sum(speeds**2)
    computed_ke = get_KE(m, speeds)
    assert computed_ke == expected_ke

# Testing MDSimulation Class
def test_MDSimulation_init(particle_A, particle_B):
    sim = MDSimulation([particle_A, particle_B])

    assert sim.n == 2
    assert sim.nsteps == 0
    assert sim.particles == [particle_A, particle_B]

# Testing MDsimulation.advance
# with collision
def test_MDSimulation_advance_with_collision(simulation, species_A, species_B):
    p1, p2 = simulation.particles

    pos_before = p1.pos.copy()
    vel_before = p1.vel.copy()
    pos_before_p2 = p2.pos.copy()
    vel_before_p2 = p2.vel.copy()

     # Advance the simulation
    dt = 0.1
    simulation.advance(dt)

    expected_pos_p1 = pos_before + vel_before * dt
    expected_pos_p2 = pos_before_p2 + vel_before_p2 * dt
    np.testing.assert_array_almost_equal(p1.pos, expected_pos_p1)
    np.testing.assert_array_almost_equal(p2.pos, expected_pos_p2)

    # Manual calculation for elastic collision
    m1, m2 = p1.mass, p2.mass
    v1_initial = vel_before
    v2_initial = vel_before_p2
    r12 = expected_pos_p1 - expected_pos_p2
    v_rel = v1_initial - v2_initial
    distance_sq = np.dot(r12, r12)
    if distance_sq == 0:
        distance_sq = 1e-10  # Avoid division by zero

    factor = np.dot(v_rel, r12) / distance_sq

    expected_vel_p1 = v1_initial - (2 * m2 / (m1 + m2)) * factor * r12
    expected_vel_p2 = v2_initial + (2 * m1 / (m1 + m2)) * factor * r12

    np.testing.assert_array_almost_equal(p1.vel, expected_vel_p1)
    np.testing.assert_array_almost_equal(p2.vel, expected_vel_p2)
    assert simulation.nsteps == 1

# MDsimulation without collision
def test_MDSimulation_advance_without_collision(simulation, species_A, species_B):
    p1 = Particle(species_A, np.array([0.2, 0.2]), np.array([1.0, 0.0]))
    p2 = Particle(species_B, np.array([0.8, 0.8]), np.array([0.0, 1.0]))
    
    sim = MDSimulation([p1, p2])
    dt = 0.1
    sim.advance(dt)
    
    expected_pos_p1 = np.array([0.3, 0.2])
    expected_pos_p2 = np.array([0.8, 0.9])
    
    np.testing.assert_array_almost_equal(p1.pos, expected_pos_p1)
    np.testing.assert_array_almost_equal(p2.pos, expected_pos_p2)
    # Velocities should remain unchanged
    np.testing.assert_array_almost_equal(p1.vel, np.array([1.0, 0.0]))
    np.testing.assert_array_almost_equal(p2.vel, np.array([0.0, 1.0]))
    
    assert sim.nsteps == 1

#collision with wall (MDSimulation boundary reflection)
def test_MDSimulation_boundary_reflection(simulation, species_A, species_B):
    # Create particles positioned to collide with walls
    p1 = Particle(species_A, np.array([0.05, 0.5]), np.array([-1.0, 0.0]))  # Left wall
    p2 = Particle(species_B, np.array([0.95, 0.5]), np.array([1.0, 0.0]))   # Right wall
    
    sim = MDSimulation([p1, p2])
    dt = 0.1
    sim.advance(dt)
    
    expected_pos_p1 = np.array([p1.radius, 0.5])  # Should be set back to radius
    expected_pos_p2 = np.array([1 - p2.radius, 0.5])  # Should be set back to 1 - radius
    expected_vel_p1 = np.array([1.0, 0.0])   # Reversed
    expected_vel_p2 = np.array([-1.0, 0.0])  # Reversed
    
    np.testing.assert_array_almost_equal(p1.pos, expected_pos_p1)
    np.testing.assert_array_almost_equal(p2.pos, expected_pos_p2)
    np.testing.assert_array_almost_equal(p1.vel, expected_vel_p1)
    np.testing.assert_array_almost_equal(p2.vel, expected_vel_p2)
    
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
    
def test_MDSimulation_reaction(species_A, species_B, species_C):
    # Create species A and B particles positioned to collide
    p1 = Particle(species_A, np.array([0.4, 0.5]), np.array([1.0, 0.0]))
    p2 = Particle(species_B, np.array([0.6, 0.5]), np.array([-1.0, 0.0]))
    
    sim = MDSimulation([p1, p2])
    dt = 0.1
    sim.advance(dt)
    
    # After advance, particles should have reacted to form a new C particle
    assert len(sim.particles) == 1  # Only species C remains
    new_p = sim.particles[0]
    assert new_p.species.name == "C"
    expected_pos = 0.5 * (p1.pos + p2.pos)
    expected_vel = (p1.mass * p1.vel + p2.mass * p2.vel) / (p1.mass + p2.mass)
    
    np.testing.assert_array_almost_equal(new_p.pos, expected_pos)
    np.testing.assert_array_almost_equal(new_p.vel, expected_vel)
    assert sim.nsteps == 1
