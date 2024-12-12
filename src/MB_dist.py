# Code from https://scipython.com/blog/the-maxwellboltzmann-distribution-in-two-dimensions/#:~:text=The%20Maxwell%E2%80%93Boltzmann%20distribution%20in%20two%20dimensions.%20Posted
# Code from https://scipython.com/blog/the-maxwellboltzmann-distribution-in-two-dimensions/#:~:text=The%20Maxwell%E2%80%93Boltzmann%20distribution%20in%20two%20dimensions.%20Posted
import os
from typing import cast, Tuple, Optional, List, Union

import matplotlib as mpl  # Aliased as per formatter's recommendation
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.text import Text
from matplotlib import patches, path
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
import numpy as np
from numpy.typing import NDArray
from numpy.random import Generator, default_rng
from scipy.spatial.distance import pdist, squareform  # type: ignore[import-untyped]

# Set Matplotlib backend based on environment variable or default to 'TkAgg'
mpl_backend: str = os.getenv("MPLBACKEND", "TkAgg")
mpl.use(mpl_backend)

# Array indices constants
X: int = 0
Y: int = 1

# static type aliases
SpeciesName = str
Color = str


class Species:
    """Defines each species to have common attributes: name, mass, radius, color"""

    def __init__(
        self, name: SpeciesName, mass: float, radius: float, color: Color
    ) -> None:
        self.name: SpeciesName = name
        self.mass: float = mass
        self.radius: float = radius
        self.color: Color = color


class Particle:
    """Defines a particle which is an instance of a species"""

    def __init__(
        self,
        species: Species,
        pos: NDArray[np.float64],
        vel: NDArray[np.float64],
    ) -> None:
        self.species: Species = species
        self.pos: NDArray[np.float64] = np.array(pos, dtype=np.float64)
        self.vel: NDArray[np.float64] = np.array(vel, dtype=np.float64)

    @property
    def mass(self) -> float:
        return self.species.mass

    @property
    def radius(self) -> float:
        return self.species.radius

    @property
    def color(self) -> Color:
        return self.species.color


class MDSimulation:
    """Molecular Dynamics simulation of particle movements and interactions in a container"""

    fig: Optional[Figure] = None  # Defined here

    def __init__(
        self,
        particles: list[Particle],
        reaction_probability: float,
        rng: Generator | None = None,
    ) -> None:
        """
        Initialize the simulation with A & B identical, circular particles of radius
        r and mass m.

        Args:
            particles: initial particle list
            reaction_probability: pseudo activation energy
            rng: random number generator
        """
        self.particles: list[Particle] = particles
        self.n: int = len(particles)
        self.nsteps: int = 0
        self.reactions: list[Tuple[Particle, Particle, Particle]] = []
        self.reaction_probability: float = reaction_probability
        self.rng: Generator = rng if rng is not None else default_rng()

    def advance(self, dt: float) -> None:
        """
        Advance the simulation by one timestep

        Args:
            dt: time increment
        """
        self.nsteps += 1

        # Update positions
        for p in self.particles:
            p.pos += p.vel * dt

        # Compute distances between all pairs
        pos_array: NDArray[np.float64] = np.array([p.pos for p in self.particles], dtype=np.float64)
        dist_matrix: NDArray[np.float64] = squareform(pdist(pos_array))

        # Prevent self collisions by setting diagonal to infinity
        np.fill_diagonal(dist_matrix, np.inf)

        # Sum of radii for each pair
        radii: NDArray[np.float64] = np.array([p.radius for p in self.particles], dtype=np.float64)
        sum_r: NDArray[np.float64] = np.add.outer(radii, radii)

        # Identify collisions (dist <= sum of radii and not zero)
        collisions: Tuple[NDArray[np.intp], NDArray[np.intp]] = cast(
            Tuple[NDArray[np.intp], NDArray[np.intp]],
            np.where((dist_matrix <= sum_r) & (dist_matrix > 0))
        )

        iarr: NDArray[np.intp] = collisions[0]
        jarr: NDArray[np.intp] = collisions[1]

        # Ensure each pair is processed only once
        k: NDArray[np.bool_] = iarr < jarr
        iarr, jarr = iarr[k], jarr[k]

        # Resolve particle collisions
        for i, j in zip(iarr, jarr):
            self.resolve_collision(self.particles[i], self.particles[j])

        # Delete A,B and add C to particles list
        if self.reactions:
            for p1, p2, new_p in self.reactions:
                if p1 in self.particles:
                    self.particles.remove(p1)
                if p2 in self.particles:
                    self.particles.remove(p2)
                self.particles.append(new_p)
            self.reactions.clear()

        # Wall collisions
        for p in self.particles:
            # Left wall
            if p.pos[X] < p.radius:
                p.pos[X] = p.radius
                p.vel[X] *= -1
            # Right wall
            elif p.pos[X] > 1 - p.radius:
                p.pos[X] = 1 - p.radius
                p.vel[X] *= -1

            # Bottom wall
            if p.pos[Y] < p.radius:
                p.pos[Y] = p.radius
                p.vel[Y] *= -1
            # Top wall
            elif p.pos[Y] > 1 - p.radius:
                p.pos[Y] = 1 - p.radius
                p.vel[Y] *= -1

    # Treats particle collision as elastic
    def resolve_collision(self, p1: Particle, p2: Particle) -> None:
        """Resolves particle collisions, dealing with AB collisions as potential reactions

        Args:
            p1: particle 1
            p2: particle 2
        """

        # If collision between A&B --> C
        if (
            (p1.species.name == "A" and p2.species.name == "B")
            or (p1.species.name == "B" and p2.species.name == "A")
        ) and (self.rng.random() < self.reaction_probability):
            new_pos: NDArray[np.float64] = (0.5 * (p1.pos + p2.pos)).astype(np.float64)
            m1: float = p1.mass
            m2: float = p2.mass
            total_mass: float = m1 + m2
            new_vel: NDArray[np.float64] = ((m1 * p1.vel + m2 * p2.vel) / total_mass).astype(np.float64)

            species_C: Species = Species(
                name="C", mass=3.0, radius=0.03, color="purple"
            )
            new_particle: Particle = Particle(species_C, new_pos, new_vel)

            # Remove A + B and add C to particles list
            self.reactions.append((p1, p2, new_particle))

        # Otherwise elastic collision
        else:
            m1_e: float = p1.mass
            m2_e: float = p2.mass
            r12: NDArray[np.float64] = cast(NDArray[np.float64], p1.pos - p2.pos)
            v12: NDArray[np.float64] = cast(NDArray[np.float64], p1.vel - p2.vel)
            r12_sq: float = np.dot(r12, r12)

            if r12_sq == 0:
                separation: float = p1.radius * 1e-3
                p1.pos[X] += separation
                p2.pos[X] -= separation
                r12 = p1.pos - p2.pos
                r12_sq = np.dot(r12, r12)

            v_rel: float = np.dot(v12, r12) / r12_sq

            p1.vel = p1.vel - (2 * m2_e / (m1_e + m2_e)) * v_rel * r12
            p2.vel = p2.vel + (2 * m1_e / (m1_e + m2_e)) * v_rel * r12


class Histogram:
    """A class to draw a Matplotlib histogram as a collection of Patches."""

    def __init__(
        self,
        data: NDArray[np.float64],
        xmax: float,
        nbars: int,
        density: bool = False,
    ) -> None:
        """
        Initialize the histogram from the data and requested bins.

        Args:
            data: histogram data points
            xmax: max histogram bin value
            nbars: number of histogram bars
            density: y/n normalize?
        """
        self.nbars: int = nbars
        self.density: bool = density
        self.bins: NDArray[np.float64] = np.linspace(0, xmax, nbars)
        self.hist: NDArray[np.float64]
        self.hist, bins = cast(
            Tuple[NDArray[np.float64], NDArray[np.float64]],
            np.histogram(data, self.bins, density=density)
        )

        # Drawing the histogram with Matplotlib patches owes a lot to
        # https://matplotlib.org/3.1.1/gallery/animation/animated_histogram.html
        # Get the corners of the rectangles for the histogram.
        self.left: NDArray[np.float64] = np.array(bins[:-1], dtype=np.float64)
        self.right: NDArray[np.float64] = np.array(bins[1:], dtype=np.float64)
        self.bottom: NDArray[np.float64] = np.zeros(len(self.left), dtype=np.float64)
        self.top: NDArray[np.float64] = self.bottom + self.hist
        nrects: int = len(self.left)
        self.nverts: int = nrects * 5
        self.verts: NDArray[np.float64] = np.zeros((self.nverts, 2), dtype=np.float64)
        self.verts[0::5, 0] = self.left
        self.verts[0::5, 1] = self.bottom
        self.verts[1::5, 0] = self.left
        self.verts[1::5, 1] = self.top
        self.verts[2::5, 0] = self.right
        self.verts[2::5, 1] = self.top
        self.verts[3::5, 0] = self.right
        self.verts[3::5, 1] = self.bottom

    def draw(self, ax: Axes) -> None:
        """
        Draw the histogram by adding appropriate patches to Axes ax.

        Args:
            ax: matplotlib axes object
        """
        codes: NDArray[np.int_] = np.ones(self.nverts, dtype=int) * path.Path.LINETO
        codes[0::5] = path.Path.MOVETO
        codes[4::5] = path.Path.CLOSEPOLY
        barpath: path.Path = path.Path(self.verts, codes)
        self.patch: PathPatch = patches.PathPatch(
            barpath, fc="tab:green", ec="k", lw=0.5, alpha=0.5
        )
        ax.add_patch(self.patch)

    def update(self, data: NDArray[np.float64]) -> None:
        """
        Update the rectangle vertices using a new histogram from data.

        Args:
            data: New data for histogram
        """
        self.hist, self.bins = np.histogram(data, self.bins, density=self.density)
        self.hist = self.hist.astype(np.float64)
        self.bins = self.bins.astype(np.float64)
        self.top = self.bottom + self.hist
        self.verts[1::5, 1] = self.top
        self.verts[2::5, 1] = self.top


def get_speeds(particles: list[Particle]) -> NDArray[np.float64]:
    """
    Gets speeds from the list of particles.

    Args:
        particles: list of particles
    Returns:
        magnitude of the (n,2) array of velocities
    """
    if not particles:
        return np.array([], dtype=np.float64)
    vel: NDArray[np.float64] = np.array([p.vel for p in particles], dtype=np.float64)
    return np.hypot(vel[:, X], vel[:, Y])


def get_KE(m: float, speeds: NDArray[np.float64]) -> float:
    """Return the total kinetic energy of all particles in scaled units.

    Args:
        m = average mass
        speeds = array of speeds
    Returns:
        total kinetic energy
    """
    return float(0.5 * m * np.sum(speeds**2))


def particle_simulator_initial_steps(
    Matrix_A: list[float] | tuple[float, float, float] | NDArray[np.float64],
    Matrix_B: list[float] | tuple[float, float, float] | NDArray[np.float64],
    Matrix_C: list[float] | tuple[float, float, float] | NDArray[np.float64],
) -> list[Particle]:
    particles: list[Particle] = []
    MATRIX_SIZE: int = 3
    error_message: str = "Matrix_A,B,C must be a list, tuple, or NumPy array with three elements: [num_A, mass_A, radius_A]"

    # Validate Matrix_A
    if not (
        len(Matrix_A) == MATRIX_SIZE
    ):
        raise ValueError(error_message)

    # Similarly validate Matrix_B and Matrix_C
    for Matrix, _name in zip(
        [Matrix_B, Matrix_C], ["Matrix_B", "Matrix_C"]
    ):
        if not (
            len(Matrix) == MATRIX_SIZE
        ):
            raise ValueError(error_message)

    # Extract properties for species A, B, C from Matrices
    num_A, mass_A, radius_A = Matrix_A
    num_B, mass_B, radius_B = Matrix_B
    num_C, mass_C, radius_C = Matrix_C

    # Define species with different properties
    species_A: Species = Species(name="A", mass=mass_A, radius=radius_A, color="red")
    species_B: Species = Species(name="B", mass=mass_B, radius=radius_B, color="blue")
    species_C: Species = Species(name="C", mass=mass_C, radius=radius_C, color="purple")

    rng: Generator = default_rng()

    # Create initial positions and velocities for each species
    # For simplicity, place species A on the left side, species B on the right, species C in the middle
    pos_A: NDArray[np.float64] = rng.random((int(num_A), 2)) * 0.4 + 0.05  # left side
    vel_A: NDArray[np.float64] = rng.random((int(num_A), 2)) - 0.5

    pos_B: NDArray[np.float64] = rng.random((int(num_B), 2)) * 0.4 + 0.55  # right side
    vel_B: NDArray[np.float64] = rng.random((int(num_B), 2)) - 0.5

    pos_C: NDArray[np.float64] = rng.random((int(num_C), 2)) * 0.4 + 0.3  # middle
    vel_C: NDArray[np.float64] = rng.random((int(num_C), 2)) - 0.5

    # Create Particle instances
    particles += (
        [Particle(species_A, p.astype(np.float64), v.astype(np.float64)) for p, v in zip(pos_A, vel_A)]
        + [Particle(species_B, p.astype(np.float64), v.astype(np.float64)) for p, v in zip(pos_B, vel_B)]
        + [Particle(species_C, p.astype(np.float64), v.astype(np.float64)) for p, v in zip(pos_C, vel_C)]
    )
    return particles


def setup_plot(
    sim: MDSimulation,
) -> Tuple[
    Figure,
    Axes,
    PathCollection,
    Axes,
    Histogram,
    Line2D,
    Line2D,
    NDArray[np.float64],
    Text,
    NDArray[np.float64],
]:
    """
    Set up the Matplotlib figures and axes for simulation and histogram.

    Args:
        sim: Molecular Dynamics Simulation
    Returns:
        tuple containing:
            - fig (Figure)
            - ax (Axes)
            - scatter (PathCollection)
            - speed_ax (Axes)
            - speed_hist (Histogram)
            - mb_line (Line2D)
            - mb_est_line (Line2D)
            - mb_est (NDArray[np.float64])
            - label (Text)
            - sgrid (NDArray[np.float64])
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", "box")
    ax.set_xticks([])
    ax.set_yticks([])

    x: list[float] = [p.pos[X] for p in sim.particles]
    y: list[float] = [p.pos[Y] for p in sim.particles]

    colors: list[Color] = [p.color for p in sim.particles]
    scatter: PathCollection = ax.scatter(x, y, c=colors, s=30)

    # The 2D Maxwell-Boltzmann distribution of speeds & Histogram setup.
    masses: list[float] = [p.mass for p in sim.particles]
    m: float = np.mean(masses)
    speeds: NDArray[np.float64] = get_speeds(sim.particles)
    mean_KE: float = float(get_KE(m, speeds) / sim.n)
    a: float = m / (2 * mean_KE)

    speed_ax: Axes = fig.add_subplot(122)
    speed_hist: Histogram = Histogram(
        speeds, xmax=np.max(speeds) * 2, nbars=50, density=True
    )
    speed_hist.draw(speed_ax)

    sgrid_hi: NDArray[np.float64] = np.linspace(0, speed_hist.bins[-1], 200)
    f: NDArray[np.float64] = 2 * a * sgrid_hi * np.exp(-a * sgrid_hi**2)
    mb_line: Line2D = speed_ax.plot(sgrid_hi, f, c="0.7")[0]

    fmax: float = np.sqrt(m / mean_KE / np.e)
    speed_ax.set_ylim(0, fmax)

    # For the distribution derived by averaging, take the abscissa speed points from the centre of the histogram bars.
    sgrid: NDArray[np.float64] = (speed_hist.bins[1:] + speed_hist.bins[:-1]) / 2
    mb_est_line: Line2D = speed_ax.plot([], [], c="r")[0]
    mb_est: NDArray[np.float64] = np.zeros(len(sgrid), dtype=np.float64)
    xlabel: float = sgrid[-1] / 2
    ylabel: float = 0.8 * fmax
    label: Text = speed_ax.text(
        xlabel, ylabel, f"$t$ = {0:.1f}s, step = {0:d}", backgroundcolor="w"
    )

    return (
        fig,
        ax,
        scatter,
        speed_ax,
        speed_hist,
        mb_line,
        mb_est_line,
        mb_est,
        label,
        sgrid,
    )

