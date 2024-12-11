import MB_dist


def caller(number_of_particles, FPS, reaction_probability):
    '''
    This is an external file to call the particle simulator.

    Inputs
        number_of_particles (int): Total number of particles in the simulation.
        FPS (int): Frames per second, determines the time step for updates.
    '''
    num_A = number_of_particles // 2
    num_B = number_of_particles - num_A 
    MB_dist.particle_simulator(num_A, num_B, FPS, reaction_probability)

caller(1000,30, 0.5)
