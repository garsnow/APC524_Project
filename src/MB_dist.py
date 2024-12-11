# Code from https://scipython.com/blog/the-maxwellboltzmann-distribution-in-two-dimensions/#:~:text=The%20Maxwell%E2%80%93Boltzmann%20distribution%20in%20two%20dimensions.%20Posted
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches, path
from matplotlib.animation import FuncAnimation
from scipy.spatial.distance import pdist, squareform
import random

X, Y = 0, 1


class Species:
    # defines each species to have common attributes: name, mass, radius, color

    def __init__(self, name, mass, radius, color):
        self.name = name
        self.mass = mass
        self.radius = radius
        self.color = color


class Particle:
    # defines a particle which is an instance of a species

    def __init__(self, species, pos, vel):
        self.species = species
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)

    @property
    def mass(self):
        return self.species.mass

    @property
    def radius(self):
        return self.species.radius

    @property
    def color(self):
        return self.species.color


class MDSimulation:
    def __init__(self, particles, reaction_probability):
        """
        Initialize the simulation with A & B identical, circular particles of radius
        r and mass m.

        """
        self.particles = particles
        self.n = len(particles)
        self.nsteps = 0
        self.reactions = []
        self.reaction_probability = reaction_probability

    def advance(self, dt):
        self.nsteps += 1
        # Update positions
        for p in self.particles:
            p.pos += p.vel * dt

        # Compute distances between all pairs
        pos_array = np.array([p.pos for p in self.particles])
        dist_matrix = squareform(pdist(pos_array))

        # Prevent self collsions by setting diagonal to infinity
        np.fill_diagonal(dist_matrix, np.inf)

        # sum of radii for each pair
        radii = np.array([p.radius for p in self.particles])
        sum_r = np.add.outer(radii, radii)

        # Identify collisions (dist <= sum of radii and not zero)
        iarr, jarr = np.where((dist_matrix <= sum_r) & (dist_matrix > 0))
        k = iarr < jarr
        iarr, jarr = iarr[k], jarr[k]

        # Resolve particle collisions
        for i, j in zip(iarr, jarr):
            self.resolve_collision(self.particles[i], self.particles[j])

        # delete A,B and add C to particles list
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

    # treats particle collision as elastic
    def resolve_collision(self, p1, p2):
        # if collision between A&B --> C
        if (p1.species.name == "A" and p2.species.name == "B") or (
            p1.species.name == "B" and p2.species.name == "A") and (
                random.random() < self.reaction_probability
        ):
            new_pos = 0.5 * (p1.pos + p2.pos)
            m1, m2 = p1.mass, p2.mass
            total_mass = m1 + m2
            new_vel = (m1 * p1.vel + m2 * p2.vel) / total_mass

            species_C = Species(name="C", mass=3.0, radius=0.03, color="purple")
            new_particle = Particle(species_C, new_pos, new_vel)

            # remove A + B and add C to particles list
            self.reactions.append((p1, p2, new_particle))

        # otherwise elastic collision
        else:
            m1, m2 = p1.mass, p2.mass
            r12 = p1.pos - p2.pos
            v12 = p1.vel - p2.vel
            r12_sq = np.dot(r12, r12)

            if r12_sq == 0:
                separation = p1.radius * 1e-3
                p1.pos[X] += separation
                p2.pos[X] -= separation
                r12 = p1.pos - p2.pos
                r12_sq = np.dot(r12, r12)

            v_rel = np.dot(v12, r12) / r12_sq

            p1.vel = p1.vel - (2 * m2 / (m1 + m2)) * v_rel * r12
            p2.vel = p2.vel + (2 * m1 / (m1 + m2)) * v_rel * r12


class Histogram:
    """A class to draw a Matplotlib histogram as a collection of Patches."""

    def __init__(self, data, xmax, nbars, density=False):
        """Initialize the histogram from the data and requested bins."""
        self.nbars = nbars
        self.density = density
        self.bins = np.linspace(0, xmax, nbars)
        self.hist, bins = np.histogram(data, self.bins, density=density)

        # Drawing the histogram with Matplotlib patches owes a lot to
        # https://matplotlib.org/3.1.1/gallery/animation/animated_histogram.html
        # Get the corners of the rectangles for the histogram.
        self.left = np.array(bins[:-1])
        self.right = np.array(bins[1:])
        self.bottom = np.zeros(len(self.left))
        self.top = self.bottom + self.hist
        nrects = len(self.left)
        self.nverts = nrects * 5
        self.verts = np.zeros((self.nverts, 2))
        self.verts[0::5, 0] = self.left
        self.verts[0::5, 1] = self.bottom
        self.verts[1::5, 0] = self.left
        self.verts[1::5, 1] = self.top
        self.verts[2::5, 0] = self.right
        self.verts[2::5, 1] = self.top
        self.verts[3::5, 0] = self.right
        self.verts[3::5, 1] = self.bottom

    def draw(self, ax):
        """Draw the histogram by adding appropriate patches to Axes ax."""
        codes = np.ones(self.nverts, int) * path.Path.LINETO
        codes[0::5] = path.Path.MOVETO
        codes[4::5] = path.Path.CLOSEPOLY
        barpath = path.Path(self.verts, codes)
        self.patch = patches.PathPatch(
            barpath, fc="tab:green", ec="k", lw=0.5, alpha=0.5
        )
        ax.add_patch(self.patch)

    def update(self, data):
        """Update the rectangle vertices using a new histogram from data."""
        self.hist, bins = np.histogram(data, self.bins, density=self.density)
        self.top = self.bottom + self.hist
        self.verts[1::5, 1] = self.top
        self.verts[2::5, 1] = self.top


def get_speeds(particles):
    """Return the magnitude of the (n,2) array of velocities, vel."""
    if not particles:
        return np.array([])
    vel = np.array([p.vel for p in particles])
    return np.hypot(vel[:, X], vel[:, Y])


def get_KE(m, speeds):
    """Return the total kinetic energy of all particles in scaled units."""
    return 0.5 * m * np.sum(speeds**2)


def particle_simulator(Matrix_A, Matrix_B, Matrix_C, time_step, reaction_probability):
    """
    Initialize and run the molecular dynamics simulation.

    Parameters:
    - Matrix_A: List or NumPy array containing [num_A, mass_A, radius_A]
    - num_B: Number of particles for species B
    - time_step: Frames per second (FPS)
    - reaction_probability: Probability of reaction upon collision
    - num_C: Number of initial particles for species C (default is 0)
    """
    # Extract properties for species A from Matrix_A and B
    num_A, mass_A, radius_A = Matrix_A
    num_B, mass_B, radius_B = Matrix_B
    num_C, mass_C, radius_C = Matrix_C

    # Define two species with different properties
    species_A = Species(name="A", mass=mass_A, radius=radius_A, color="red")
    species_B = Species(name="B", mass=mass_B, radius=mass_B, color="blue")
    species_C = Species(name="C", mass=mass_C, radius=mass_C, color="purple")

    # Create initial positions and velocities for each species
    # For simplicity, place species A on the left side, species B on the right
    pos_A = np.random.rand(int(num_A), 2) * 0.4 + 0.05  # left side
    vel_A = np.random.rand(int(num_A), 2) - 0.5

    pos_B = np.random.rand(int(num_B), 2) * 0.4 + 0.55  # right side
    vel_B = np.random.rand(int(num_B), 2) - 0.5

    pos_C = np.random.rand(int(num_C), 2) * 0.4 # middle
    vel_C = np.random.rand(int(num_C), 2) - 0.5

    particles = (
        [Particle(species_A, p, v) for p, v in zip(pos_A, vel_A)]
        + [Particle(species_B, p, v) for p, v in zip(pos_B, vel_B)]
        + [Particle(species_C, p, v) for p, v in zip(pos_C, vel_C)]
    )

    sim = MDSimulation(particles, reaction_probability)

    FPS = time_step
    dt = 1 / FPS

    # Set up plotting
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", "box")
    ax.set_xticks([])
    ax.set_yticks([])

    # We'll use a scatter plot so we can easily set individual particle colors
    x = [p.pos[X] for p in sim.particles]
    y = [p.pos[Y] for p in sim.particles]
    colors = [p.color for p in sim.particles]
    scatter = ax.scatter(x, y, c=colors, s=30)

    # The 2D Maxwell-Boltzmann equilibrium distribution of speeds.

    # caclulates an average mass for the two particles
    masses = [p.mass for p in sim.particles]
    m = np.mean(masses)

    # computes speeds from particles
    speeds = get_speeds(sim.particles)

    mean_KE = get_KE(m, speeds) / sim.n
    a = m / 2 / mean_KE

    # speed subplot
    speed_ax = fig.add_subplot(122)
    speed_hist = Histogram(speeds, xmax=np.max(speeds) * 2, nbars=50, density=True)
    speed_hist.draw(speed_ax)

    # Use a high-resolution grid of speed points so that the exact distribution
    # looks smooth.
    sgrid_hi = np.linspace(0, speed_hist.bins[-1], 200)
    f = 2 * a * sgrid_hi * np.exp(-a * sgrid_hi**2)
    (mb_line,) = speed_ax.plot(sgrid_hi, f, c="0.7")

    # Maximum value of the 2D Maxwell-Boltzmann speed distribution.
    fmax = np.sqrt(m / mean_KE / np.e)
    speed_ax.set_ylim(0, fmax)

    # For the distribution derived by averaging, take the abcissa speed points from
    # the centre of the histogram bars.
    sgrid = (speed_hist.bins[1:] + speed_hist.bins[:-1]) / 2
    (mb_est_line,) = speed_ax.plot([], [], c="r")
    mb_est = np.zeros(len(sgrid))

    # A text label indicating the time and step number for each animation frame.
    xlabel, ylabel = sgrid[-1] / 2, 0.8 * fmax
    label = speed_ax.text(
        xlabel, ylabel, f"$t$ = {0:.1f}s, step = {0:d}", backgroundcolor="w"
    )

    def init_anim():
        """Initialize the animation"""
        scatter.set_offsets(np.zeros((0, 2)))
        return scatter, label

    def animate(i):
        """Advance the animation by one step and update the frame."""

        nonlocal mb_est
        sim.advance(dt)

        x = [p.pos[X] for p in sim.particles]
        y = [p.pos[Y] for p in sim.particles]
        colors = [p.color for p in sim.particles]
        scatter.set_facecolors(colors)
        scatter.set_offsets(np.column_stack((x, y)))

        speeds = get_speeds(sim.particles)
        speed_hist.update(speeds)

        # Once the simulation has approached equilibrium a bit, start averaging
        # the speed distribution to indicate the approximation to the Maxwell-
        # Boltzmann distribution.
        if i >= IAV_START:
            mb_est += (speed_hist.hist - mb_est) / (i - IAV_START + 1)
            mb_est_line.set_data(sgrid, mb_est)

        label.set_text(f"$t$ = {i * dt:.1f} ns, step = {i:d}")

        return scatter, speed_hist.patch, mb_est_line, label

    # Only start averaging the speed distribution after frame number IAV_ST.
    IAV_START = 1000
    # Number of frames; set to None to run until explicitly quit.
    frames = 1000
    anim = FuncAnimation(
        fig, animate, frames=frames, interval=10, blit=False, init_func=init_anim
    )

    anim.save("MB_simulation.gif", writer="Pillow")
    plt.show()
