import MB_dist


def caller(Matrix_A, Matrix_B, Matrix_C, FPS, reaction_probability):
    """
    This is an external file to call the particle simulator.

    Inputs
        Matrix_A,B,C: For each individual particle type, specifies the
                      initial number, mass, and radius
        FPS (int): Frames per second, determines the time step for updates.
        reaction_probability: When A and B collide, how often do they react to form C
    """
    MB_dist.particle_simulator(Matrix_A, Matrix_B, Matrix_C, FPS, reaction_probability)


# Matrix template: Matrix_A = [number of initial A particles, A mass, A radius]
Matrix_A = [500, 1.0, 0.01]
Matrix_B = [500, 2.0, 0.02]
Matrix_C = [0, 3.0, 0.03]
caller(Matrix_A, Matrix_B, Matrix_C, 30, 0.5)
