import torch

from trainer.FedAvg.server import Server as Base_Server
import copy
from sklearn.preprocessing import normalize
import torch.nn.functional as F
from utils.loss import DiversityLoss
from torch.optim import *

import numpy as np

from utils.nets import TwinBranchNets

MIN_SAMPLES_PER_LABEL = 1


class Server(Base_Server):
    def __init__(self, generator, gen_epochs, gen_lr=1e-4, gen_alpha=1, gen_beta=0, gen_eta=1,
                 distill_temperature=20.,
                 **kwargs):
        super(Server, self).__init__(**kwargs)
        self.algorithm_name = "FedGen"

        assert isinstance(self.model, TwinBranchNets), \
            "FedGen need model in format of [feature_extractor, classifier]. Now, only TwinBranchNets is ok."

        # config generator
        self.generator = generator
        self.generator.to(self.device)

        self.gen_epochs = gen_epochs
        gen_optim_kwargs = copy.deepcopy(self.optim_kwargs)
        gen_optim_kwargs['lr'] = gen_lr
        self.gen_optimizer = eval(self.opt_name)(self.generator.parameters(), **gen_optim_kwargs)

        self.diversity_loss = DiversityLoss(metric='l1')

        self.distill_temperature = distill_temperature

        # Hyperparameter of updating generator, same as ensemble_alpha, ensemble_beta, and
        # ensemble_eta in FedGen project 'utils.model_config.RUNCONFIGS'. In FedGen project,
        # the value of them is 1, 0, 1 for all. Here, we take their default value as: (1, 0, 1).
        self.gen_alpha = gen_alpha
        self.gen_beta = gen_beta
        self.gen_eta = gen_eta

    def distribute_model(self):
        # distribute model
        Base_Server.distribute_model(self)
        # distribute generator
        for client in self.selected_clients: client.generator = self.generator

    def aggregate(self):
        """aggregate clients' classifier"""
        # 1. update generator
        self.update_generator()

        # 2. get classifier of clients
        Base_Server.aggregate(self)

    def update_generator(self):
        self.generator.to(self.device)
        self.model.to(self.device)

        self.generator.train()
        self.model.eval()

        # get clients' distribution
        c_k = [client.label_counts for client in self.selected_clients]
        label_weights, qualified_labels = self.get_label_weights(c_k)

        # get clients' classifier
        client_classifier = [client.model.classifier for client in self.selected_clients]
        for model in client_classifier:
            model.to(self.device)
            model.eval()

        for epoch in range(self.gen_epochs):
            # print(f"update server generator, {epoch} epoch.")
            # sample a batch of generator's input
            batch_labels = np.random.choice(qualified_labels, self.batch_size)
            y = torch.tensor(batch_labels, dtype=torch.int64).to(self.device)

            # 1. calculate diversity loss
            z, eps = self.generator(y)
            div_loss = self.diversity_loss(z, eps)

            # 2. teacher loss (loss in real label  y and clients' output logits)
            teacher_loss = 0
            teacher_output = 0
            # use clients' model to cal logits of generated data
            for idx, client_model in enumerate(client_classifier):
                # cal teacher loss as in Equation (4)
                client_output = client_model(z)
                weight = label_weights[batch_labels][:, idx].reshape(-1, 1)
                weight_tensor = torch.tensor(weight, dtype=torch.float32).to(self.device)

                loss_temp = self.loss_fn(client_output, y)
                teacher_loss += torch.mean(loss_temp * weight_tensor)

                # cal teacher logits for student loss
                # expand_weight = np.tile(weight, (1, self.unique_labels))
                expand_weight = np.tile(weight, (1, self.num_classes))
                expand_weight_tensor = torch.tensor(expand_weight, dtype=torch.float32).to(self.device)
                teacher_output += client_output * expand_weight_tensor

            # 3. student_loss
            student_output = self.model.classifier(z)
            student_loss = F.kl_div(F.softmax(student_output / self.distill_temperature, dim=-1),
                                    F.softmax(teacher_output / self.distill_temperature, dim=-1),
                                    reduction='batchmean')

            loss = self.gen_alpha * teacher_loss + self.gen_beta * student_loss + \
                   self.gen_eta * div_loss

            # backward & step optim
            self.gen_optimizer.zero_grad()
            loss.backward()
            self.gen_optimizer.step()

        self.generator.eval()

    @staticmethod
    def get_label_weights(c_k):
        c_k_array = np.array(c_k).T
        label_weights = normalize(c_k_array, axis=1, norm='l1')
        qualified_labels = np.where(np.max(c_k_array, axis=1) > MIN_SAMPLES_PER_LABEL)[0]
        return label_weights, qualified_labels
