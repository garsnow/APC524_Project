import sys
from pathlib import Path
from typing import List, Tuple, Optional, cast

import numpy as np
from numpy.random import Generator, default_rng
import pytest

from src.MB_dist import Histogram, MDSimulation, Particle, Species, get_KE, get_speeds
from numpy.typing import NDArray

# Add 'src/' directory to sys.path
src_path: Path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path.resolve()))

# Initialize the random number generator
rng: Generator = np.random.default_rng()


X: int = 0
Y: int = 1


@pytest.fixture
def species_A() -> Species:
    return cast(Species(name="A", mass=1.0, radius=0.05, color="red"))


@pytest.fixture
def species_B() -> Species:
    return Species(name="B", mass=2.0, radius=0.05, color="blue")


@pytest.fixture
def species_C() -> Species:
    return Species(name="C", mass=3.0, radius=0.03, color="purple")


@pytest.fixture
def particle_A(species_A: Species) -> Particle:
    pos: NDArray[np.float_] = np.array([0.2, 0.5], dtype=np.float64)
    vel: NDArray[np.float_] = np.array([1.0, 0.0], dtype=np.float64)
    return Particle(species_A, pos, vel)


@pytest.fixture
def particle_B(species_B: Species) -> Particle:
    pos: NDArray[np.float_] = np.array([0.8, 0.5], dtype=np.float64)
    vel: NDArray[np.float_] = np.array([-1.0, 0.0], dtype=np.float64)
    return Particle(species_B, pos, vel)


@pytest.fixture
def particle_C(species_C: Species) -> Particle:
    pos: NDArray[np.float_] = np.array([0.5, 0.5], dtype=np.float64)
    vel: NDArray[np.float_] = np.array([0.0, 0.0], dtype=np.float64)
    return Particle(species_C, pos, vel)


@pytest.fixture
def simulation(particle_A: Particle, particle_B: Particle) -> MDSimulation:
    particles: List[Particle] = [particle_A, particle_B]
    reaction_probability: float = 0.0
    return MDSimulation(particles, reaction_probability)


@pytest.fixture
def simulation_reaction(species_A: Species, species_B: Species) -> MDSimulation:
    # Create two particles positioned to collide after dt=0.1
    p1: Particle = Particle(species_A, np.array([0.45, 0.5], dtype=np.float64), np.array([1.0, 0.0], dtype=np.float64))
    p2: Particle = Particle(species_B, np.array([0.55, 0.5], dtype=np.float64), np.array([-1.0, 0.0], dtype=np.float64))
    reaction_probability: float = 1.0  # always react
    return MDSimulation([p1, p2], reaction_probability)


@pytest.fixture
def simulation_elastic(species_A: Species) -> MDSimulation:
    # Create two particles of the same species to test elastic collision
    p1: Particle = Particle(species_A, np.array([0.45, 0.5], dtype=np.float64), np.array([1.0, 0.0], dtype=np.float64))
    p2: Particle = Particle(species_A, np.array([0.55, 0.5], dtype=np.float64), np.array([-1.0, 0.0], dtype=np.float64))
    reaction_probability: float = 0.0  # no reaction
    return MDSimulation([p1, p2], reaction_probability)


# Testing the get_speeds function
@pytest.mark.parametrize(
    ("particles", "expected_speeds"),
    [
        # Basic positive velocities
        (
            [
                Particle(
                    Species("A", 1.0, 0.01, "red"),
                    np.array([0.2, 0.2], dtype=np.float64),
                    np.array([3.0, 4.0], dtype=np.float64),
                ),
                Particle(
                    Species("B", 2.0, 0.02, "blue"),
                    np.array([0.3, 0.3], dtype=np.float64),
                    np.array([0.0, 0.0], dtype=np.float64),
                ),
                Particle(
                    Species("A", 1.0, 0.01, "red"),
                    np.array([0.4, 0.4], dtype=np.float64),
                    np.array([1.0, 1.0], dtype=np.float64),
                ),
            ],
            np.array([5.0, 0.0, np.sqrt(2)], dtype=np.float64),
        ),
        # Mixed negative, positive, zero velocities
        (
            [
                Particle(
                    Species("A", 1.0, 0.01, "red"),
                    np.array([0.2, 0.2], dtype=np.float64),
                    np.array([-3.0, -4.0], dtype=np.float64),
                ),
                Particle(
                    Species("B", 2.0, 0.02, "blue"),
                    np.array([0.3, 0.3], dtype=np.float64),
                    np.array([0.0, -2.0], dtype=np.float64),
                ),
                Particle(
                    Species("A", 1.0, 0.01, "red"),
                    np.array([0.4, 0.4], dtype=np.float64),
                    np.array([3.0, -1.0], dtype=np.float64),
                ),
            ],
            np.array([5.0, 2.0, np.sqrt(10)], dtype=np.float64),
        ),
        # Zero velocities
        (
            [
                Particle(
                    Species("A", 1.0, 0.01, "red"),
                    np.array([0.2, 0.2], dtype=np.float64),
                    np.array([0.0, 0.0], dtype=np.float64),
                )
            ],
            np.array([0.0], dtype=np.float64),
        ),
        # Large and small velocities
        (
            [
                Particle(
                    Species("A", 1.0, 0.01, "red"),
                    np.array([0.2, 0.2], dtype=np.float64),
                    np.array([1e6, 1e6], dtype=np.float64),
                ),
                Particle(
                    Species("B", 2.0, 0.02, "blue"),
                    np.array([0.3, 0.3], dtype=np.float64),
                    np.array([1e-10, 1e-10], dtype=np.float64),
                ),
            ],
            np.array([np.sqrt(2) * 1e6, np.sqrt(2) * 1e-10], dtype=np.float64),
        ),
    ],
)
def test_get_speeds(particles: List[Particle], expected_speeds: NDArray[np.float64]) -> None:
    computed_speeds: NDArray[np.float64] = get_speeds(particles).astype(np.float64)
    np.testing.assert_array_almost_equal(computed_speeds, expected_speeds)


# Test with an empty list of particles
def test_get_speeds_empty() -> None:
    particles: List[Particle] = []
    expected_speeds: NDArray[np.float64] = np.array([], dtype=np.float64)

    computed_speeds: NDArray[np.float64] = get_speeds(particles).astype(np.float64)
    np.testing.assert_array_equal(computed_speeds, expected_speeds)


# Testing the get_KE function
def test_get_KE() -> None:
    species_A: Species = Species(name="A", mass=1.0, radius=0.01, color="red")
    species_B: Species = Species(name="B", mass=2.0, radius=0.02, color="blue")

    particle_A: Particle = Particle(species_A, np.array([0.1, 0.1], dtype=np.float64), np.array([3.0, 4.0], dtype=np.float64))
    particle_B: Particle = Particle(species_B, np.array([0.9, 0.9], dtype=np.float64), np.array([0.0, 0.0], dtype=np.float64))

    speeds: NDArray[np.float64] = get_speeds([particle_A, particle_B])
    m: float = 2.0  # Average mass or specific mass as per your simulation logic
    expected_ke: float = 0.5 * m * np.sum(speeds**2)
    computed_ke: float = get_KE(m, speeds)
    assert computed_ke == expected_ke


# Testing MDSimulation Class Initialization
def test_MDSimulation_init(particle_A: Particle, particle_B: Particle) -> None:
    reaction_probability: float = 0.0
    sim: MDSimulation = MDSimulation([particle_A, particle_B], reaction_probability)
    number_of_particles: int = 2
    number_steps: int = 0
    assert sim.n == number_of_particles
    assert sim.nsteps == number_steps
    assert sim.particles == [particle_A, particle_B]
    assert sim.reaction_probability == reaction_probability
    assert sim.reactions == []


# Testing MDSimulation.advance with collision
def test_MDSimulation_advance_with_collision(simulation_elastic: MDSimulation) -> None:
    p1: Particle
    p2: Particle
    p1, p2 = simulation_elastic.particles

    pos_before: NDArray[np.float64] = p1.pos.copy()
    vel_before: NDArray[np.float64] = p1.vel.copy()
    pos_before_p2: NDArray[np.float64] = p2.pos.copy()
    vel_before_p2: NDArray[np.float64] = p2.vel.copy()

    # Advance the simulation
    dt: float = 0.1
    simulation_elastic.advance(dt)

    expected_pos_p1: NDArray[np.float64] = pos_before + vel_before * dt
    expected_pos_p2: NDArray[np.float64] = pos_before_p2 + vel_before_p2 * dt

    expected_vel_p1: NDArray[np.float64] = vel_before_p2  # [-1.0, 0.0]
    expected_vel_p2: NDArray[np.float64] = vel_before      # [1.0, 0.0]

    np.testing.assert_array_almost_equal(p1.pos, expected_pos_p1)
    np.testing.assert_array_almost_equal(p2.pos, expected_pos_p2)
    np.testing.assert_array_almost_equal(p1.vel, expected_vel_p1)
    np.testing.assert_array_almost_equal(p2.vel, expected_vel_p2)

    number_steps: int = 1
    assert simulation_elastic.nsteps == number_steps


# Testing MDSimulation.advance without collision
def test_MDSimulation_advance_without_collision(species_A: Species, species_B: Species) -> None:
    p1: Particle = Particle(species_A, np.array([0.2, 0.2], dtype=np.float64), np.array([1.0, 0.0], dtype=np.float64))
    p2: Particle = Particle(species_B, np.array([0.8, 0.8], dtype=np.float64), np.array([0.0, 1.0], dtype=np.float64))

    reaction_probability: float = 0.0
    sim: MDSimulation = MDSimulation([p1, p2], reaction_probability)
    dt: float = 0.1
    sim.advance(dt)

    expected_pos_p1: NDArray[np.float64] = np.array([0.3, 0.2], dtype=np.float64)
    expected_pos_p2: NDArray[np.float64] = np.array([0.8, 0.9], dtype=np.float64)

    np.testing.assert_array_almost_equal(p1.pos, expected_pos_p1)
    np.testing.assert_array_almost_equal(p2.pos, expected_pos_p2)
    # Velocities should remain unchanged
    np.testing.assert_array_almost_equal(p1.vel, np.array([1.0, 0.0], dtype=np.float64))
    np.testing.assert_array_almost_equal(p2.vel, np.array([0.0, 1.0], dtype=np.float64))

    number_steps: int = 1
    assert sim.nsteps == number_steps


# Testing MDSimulation boundary reflection
def test_MDSimulation_boundary_reflection(species_A: Species, species_B: Species) -> None:
    # Create particles positioned to collide with walls
    p1: Particle = Particle(species_A, np.array([0.05, 0.5], dtype=np.float64), np.array([-1.0, 0.0], dtype=np.float64))  # Left wall
    p2: Particle = Particle(species_B, np.array([0.95, 0.5], dtype=np.float64), np.array([1.0, 0.0], dtype=np.float64))   # Right wall

    reaction_probability: float = 0.0
    sim: MDSimulation = MDSimulation([p1, p2], reaction_probability)
    dt: float = 0.1
    sim.advance(dt)

    expected_pos_p1: NDArray[np.float64] = np.array([p1.radius, 0.5], dtype=np.float64)  # Should be set back to radius
    expected_pos_p2: NDArray[np.float64] = np.array([1 - p2.radius, 0.5], dtype=np.float64)  # Should be set back to 1 - radius
    expected_vel_p1: NDArray[np.float64] = np.array([1.0, 0.0], dtype=np.float64)  # Reversed
    expected_vel_p2: NDArray[np.float64] = np.array([-1.0, 0.0], dtype=np.float64)  # Reversed

    np.testing.assert_array_almost_equal(p1.pos, expected_pos_p1)
    np.testing.assert_array_almost_equal(p2.pos, expected_pos_p2)
    np.testing.assert_array_almost_equal(p1.vel, expected_vel_p1)
    np.testing.assert_array_almost_equal(p2.vel, expected_vel_p2)

    number_steps: int = 1
    assert sim.nsteps == number_steps


# Testing Histogram class initialization
def test_Histogram_init() -> None:
    data: NDArray[np.float64] = np.array([1, 2, 2, 3, 3, 3], dtype=np.float64)
    xmax: float = 4.0
    nbars: int = 3
    density: bool = False

    hist_obj: Histogram = Histogram(data, xmax, nbars, density)

    expected_hist: NDArray[np.float64]
    expected_bins: NDArray[np.float64]
    expected_hist, expected_bins = np.histogram(
        data, bins=np.linspace(0, 4, 3), density=False
    )
    assert np.array_equal(hist_obj.hist, expected_hist)
    assert np.array_equal(hist_obj.bins, expected_bins)
    assert hist_obj.nbars == nbars
    assert hist_obj.density == density


# Testing Histogram.update method
def test_Histogram_update() -> None:
    data_initial: NDArray[np.float64] = np.array([1, 2, 2, 3, 3, 3], dtype=np.float64)
    xmax: float = 4.0
    nbars: int = 3
    density: bool = False

    hist_obj: Histogram = Histogram(data_initial, xmax, nbars, density)
    data_new: NDArray[np.float64] = np.array([2, 3, 4, 4, 4, 4], dtype=np.float64)
    hist_obj.update(data_new)

    expected_hist, expected_bins = cast(
        Tuple[NDArray[np.float64], NDArray[np.float64]],
        np.histogram(data_new, bins=np.linspace(0, 4, 3), density=False)
    )
    assert np.array_equal(hist_obj.hist, expected_hist)
    expected_top: NDArray[np.float64] = np.zeros(len(expected_hist), dtype=np.float64) + expected_hist
    np.testing.assert_array_almost_equal(hist_obj.top, expected_top)


def test_MDSimulation_reaction(simulation_reaction: MDSimulation) -> None:
    initial_num_particles: int = len(simulation_reaction.particles)
    particle_before_reaction: int = 2
    assert (
        initial_num_particles == particle_before_reaction
    ), "There should be two particles before reaction."
    p1: Particle
    p2: Particle
    p1, p2 = simulation_reaction.particles

    # Advance the simulation
    dt: float = 0.1
    simulation_reaction.advance(dt)

    # After advance, particles should have reacted to form a new C particle
    expected_num_particles: int = initial_num_particles - 2 + 1
    assert (
        len(simulation_reaction.particles) == expected_num_particles
    ), "Number of particles after reaction is incorrect."
    new_p: Particle = simulation_reaction.particles[-1]
    assert new_p.species.name == "C", "New particle should be of species 'C'."
    expected_pos: NDArray[np.float64] = cast(
        NDArray[np.float64], 
        (0.5 * (p1.pos + p2.pos))
    )
    expected_vel: NDArray[np.float64] = cast(
        NDArray[np.float64],
        (p1.mass * p1.vel + p2.mass * p2.vel) / (p1.mass + p2.mass)
    )
    np.testing.assert_array_almost_equal(new_p.pos, expected_pos)
    np.testing.assert_array_almost_equal(new_p.vel, expected_vel)

    number_steps: int = 1
    assert simulation_reaction.nsteps == number_steps, "Number of simulation steps is incorrect."

