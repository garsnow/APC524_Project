import MB_dist

caller(1000,30,1)

def caller(number_of_particles, FPS, particle_mass):
    '''
    This is an external file to call the particle simulator.

    Inputs
        number_of_particles(int): how many particles will be reacting in the box.
        FPS(int): frames per second, the inverse of the time step that updates simulation.
        particle_mass(int): mass of every particle in the simulation.
    '''
    MB_dist.particle_simulator(number_of_particles, FPS, particle_mass)

