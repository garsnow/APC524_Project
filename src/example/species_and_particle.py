class Species: 
    #defines each species to have common attributes: name, mass, radius, color
    def __init__(self, name, mass, radius, color):
        self.name = name
        self.mass = mass
        self.radius = radius
        self.color = color

class Particle: 
    #defines a particle which is an instance of a species
    def __init__(self, species, pos, vel)
    self.species = species 
    self.pos = np.array(pos, dtype = float)
    self.vel = np.array(vel, dtype = float)

    @property
    def mass(self):
        return self.species.mass

    @property
    def radius(self):
        return self.species.radius

    @property
    def color(self):
        return self.species.color
