import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.input_size = kwargs.get("input_size", 2)
        self.loss_function = kwargs.get("loss_function", nn.BCELoss())
        self._optimizer = None

        self.model = nn.Sequential(
            nn.Linear(self.input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        output = self.model(x)
        return output

    def generate(self, sample_size):
        latent_space_samples = torch.randn((sample_size, self.input_size))
        return self.__call__(latent_space_samples)

    def generate_latent_samples(self, batch_size):
        return torch.randn((batch_size, self.input_size))

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, new_optimizer):
        self._optimizer = new_optimizer

    def partial_train(self, output_discriminator_generated):
        self.zero_grad()
        real_samples_labels = torch.ones((output_discriminator_generated.size()[0], 1))
        loss_generator = self.loss_function(
            output_discriminator_generated, real_samples_labels
        )
        loss_generator.backward()
        self.optimizer.step()


if __name__ == "__main__":
    """ test generator """
    input_size = 2
    generator = Generator(input_size=input_size)

    res = generator.generate(sample_size=100)
    print(res.size())
    generator.optimizer = torch.optim.Adam(generator.parameters())
    print(generator.parameters)
    print(generator.optimizer)
