import MB_dist


def caller(number_of_particles, FPS):
    '''
    This is an external file to call the particle simulator.

    Inputs
        number_of_particles(int): how many particles will be reacting in the box.
        FPS(int): frames per second, the inverse of the time step that updates simulation.
        particle_mass(int): mass of every particle in the simulation.
    '''
    num_A = number_of_particles // 2
    num_B = number_of_particles - num_A 
    MB_dist.particle_simulator(num_A, num_B, FPS)

caller(1000,30)
