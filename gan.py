import torch


class Gan:
    def __init__(self, config=None):
        self.config = dict(
            num_epoch=10,
            learning_rate=0.001,
            batch_size=32,
            shuffle=True
        )
        if config:
            self.config.update(config)

        self.generator = None
        self.discriminator = None

    def set_config(self, **kwargs):
        self.config.update(kwargs)

    def set_generator(self, generator, optimizer=torch.optim.Adam):
        self.generator = generator
        self.generator.optimizer = optimizer(
            self.generator.parameters(),
            lr=self.config["learning_rate"]
        )

    def set_discriminator(self, discriminator, optimizer=torch.optim.Adam):
        self.discriminator = discriminator
        self.discriminator.optimizer = optimizer(
            self.discriminator.parameters(),
            lr=self.config["learning_rate"]
        )

    def load_train(self, train_set):
        return torch.utils.data.DataLoader(
            train_set, batch_size=self.config["batch_size"], shuffle=self.config["shuffle"]
            )

    def train(self, train_set):
        assert self.generator
        assert self.discriminator

        train_loader = self.load_train(train_set)

        for epoch in range(self.config["num_epoch"]):
            for batch_index, (real_samples, _) in enumerate(train_loader):
                # Data for training D
                generated_samples = self.generator.generate(self.config["batch_size"])
                all_samples = torch.cat((real_samples, generated_samples))

                # Labels of data + generated samples for D
                real_samples_labels = torch.ones((self.config["batch_size"], 1))
                generated_samples_labels = torch.zeros((self.config["batch_size"], 1))
                all_samples_labels = torch.cat(
                    (real_samples_labels, generated_samples_labels)
                )

                # Training D
                self.discriminator.partial_train(all_samples, all_samples_labels)

                # Training G
                generated_samples = self.generator.generate(self.config["batch_size"])
                output_discriminator_generated = self.discriminator.predict(generated_samples)
                self.generator.partial_train(output_discriminator_generated)

    # def output_tensorboard(self):
    #     from torch.utils.tensorboard import SummaryWriter
    #     writer = SummaryWriter()
    #     writer.add_scalar("Loss D/train", loss_discriminator, epoch)
    #     writer.add_scalar("Loss G/train", loss_generator, epoch)
    #     writer.add_histogram('Generated x1 distribution', generated_samples[:, 0], epoch)
    #
    #     writer.close()

    # def output_print(self, func):
    #     def wrapper(*args, **kwargs):
    #         if epoch % 10 == 0 and n == batch_size - 1:
    #             func(*args, **kwargs)
    #             print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
    #             print(f"Epoch: {epoch} Loss G.: {loss_generator}")
    #     return wrapper

