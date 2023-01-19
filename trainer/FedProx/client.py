import copy
from utils.base_train import train_model
from trainer.FedAvg.client import Client as BaseClient


class Client(BaseClient):
    """
    Client for FedProx

    Args:
        mu: parameter for proximal. According to FedProx project, the value of mu can be decided by
            selecting from candidate set {0.001, 0.01, 0.1, 1}. In their paper, the value of mu is
            1, 1, 1, 0.001, and 0.01 for Synthetic, MNIST, FEMNIST, Shakespare, Sent140, respectively.
    """

    def __init__(self, mu: float = 0.01, **config):
        super(Client, self).__init__(**config)
        self.mu = mu
        self.global_model = None

    """In client, the only difference is update with proximal term"""
    def update(self, epochs=1, verbose=0):
        # memory the server's model
        self.global_model = copy.deepcopy(self.model)
        self.global_model.to(self.device)

        train_model(self.model, self.train_loader, self.optimizer,
                    self.fed_prox_loss, epochs, self.device, verbose)

        # release cuda resources
        self.global_model.to('cpu')

    def fed_prox_loss(self, log_probs, labels):
        proximal_term = 0.0
        # iterate through the current and global model parameters
        for w, w_t in zip(self.model.parameters(), self.global_model.parameters()):
            # update the proximal term
            # proximal_term += torch.sum(torch.abs((w-w_t)**2))
            proximal_term += (w - w_t).norm(2)
        loss = self.loss_fn(log_probs, labels) + (self.mu / 2) * proximal_term
        return loss
