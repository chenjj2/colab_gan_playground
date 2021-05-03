from discriminator import Discriminator
from generator import Generator
from gan import Gan
from generate_sample import create_data


data = create_data()
G = Generator()
D = Discriminator()

gan = Gan()
gan.set_discriminator(D)
gan.set_generator(G)

gan.train(data)

import pdb; pdb.set_trace()



