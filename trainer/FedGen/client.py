import torch
import numpy as np
import torch.nn.functional as F

from utils import train_model
from utils.nets import TwinBranchNets
from ..FedAvg.client import Client as BaseClient


class Client(BaseClient):
    def __init__(self, generative_alpha=10.0, generative_beta=1.0, distill_temperature=20., **kwargs):
        super(Client, self).__init__(**kwargs)
        assert isinstance(self.model, TwinBranchNets), \
            "FedGen need model in format of [feature_extractor, classifier]. Now, only TwinBranchNets is ok."

        # count label of client
        self.label_counts = [0 for _ in range(self.num_classes)]
        self.available_labels = []
        self.init_label_counts()

        # generator for distill
        self.generator = None

        # Generative alpha and beta, will be updated by server according to global iteration.
        # In original FedGen project 'utils.model_config.RUNCONFIGS' the value of generative_alpha
        # and generative_beta is 10, 10 for mnist and celeb， 10, 1 for emnist.  Here, we take
        # them same as emnist: (10, 1).
        self.generative_alpha = generative_alpha
        self.generative_beta = generative_beta
        self.distill_temperature = distill_temperature

        self.glob_iter = 0

    def init_label_counts(self):
        self.model.to(self.device)
        self.model.eval()

        # count num of samples for each label in the dataset
        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            for i in range(self.num_classes):
                idx = torch.nonzero(y == i).view(-1)
                self.label_counts[i] += len(idx)

        # get available labels
        for i in range(self.num_classes):
            if self.label_counts[i] > 0:
                self.available_labels.append(i)

    def update(self, epochs=1, verbose=0):
        # In original FedGen project, the early_stop is fixed 20, thus we use 20 as default.
        # 0-20 epoch, use fedgen_client_loss

        # when glob_iter = 0, use normal local update.
        train_model(self.model, self.train_loader, self.optimizer,
                    self.fedgen_loss_fn if self.glob_iter > 0 else self.loss_fn,
                    epochs, self.device, verbose)

    def fedgen_loss_fn(self, output, targets):
        """Clients' local update loss fn of FedGen.

        See original FedGen project 'FLAlgorithm.users.userpFedGen' for details
        """

        # update generative_alpha and generative_beta according to glob_iter
        generative_alpha = hyperparameter_scheduler(self.glob_iter, decay=0.98, init_lr=self.generative_alpha)
        generative_beta = hyperparameter_scheduler(self.glob_iter, decay=0.98, init_lr=self.generative_beta)

        # step 1. calculate predictive loss
        predictive_loss = self.loss_fn(output, targets)

        # step 2. calculate user latent loss
        # i.e. kl divergence of output in x and output in z
        z, _ = self.generator(targets)
        y_gen = self.model.classifier(z)
        user_latent_loss = F.kl_div(F.softmax(output / self.distill_temperature, dim=-1),
                                    F.softmax(y_gen / self.distill_temperature, dim=-1),
                                    reduction='batchmean')

        # step 3. calculate teacher loss
        sampled_y = np.random.choice(self.available_labels, targets.size(0))
        sampled_y = torch.tensor(sampled_y, dtype=torch.int64).to(self.device)

        z, _ = self.generator(sampled_y)
        y_gen = self.model.classifier(z)

        teacher_loss = self.loss_fn(y_gen, sampled_y)

        loss_value = predictive_loss + generative_alpha * teacher_loss + generative_beta * user_latent_loss
        return loss_value


def hyperparameter_scheduler(glob_iter, decay=0.98, init_lr=0.1, lr_decay_epoch=1):
    """Decay learning rate by a factor of 0.95 every lr_decay_epoch epochs."""
    lr = max(1e-4, init_lr * (decay ** (glob_iter // lr_decay_epoch)))
    return lr
