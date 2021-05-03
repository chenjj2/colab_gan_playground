from torch import nn


class Discriminator(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.loss_function = kwargs.get("loss_function", nn.BCELoss())
        self._optimizer = None

        self.model = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.model(x)
        return output

    def predict(self, samples):
        return self.__call__(samples)

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, new_optimizer):
        self._optimizer = new_optimizer

    def partial_train(self, samples, samples_labels):
        self.zero_grad()
        predictions = self.predict(samples)
        loss = self.loss_function(predictions, samples_labels)
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    """ test discriminator """
    discriminator = Discriminator()
