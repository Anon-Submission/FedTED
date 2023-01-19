import copy

import torch

from trainer.FedAvg.server import Server as BaseServer

import numpy as np
import torch.nn.functional as F
from torch.optim import *
from utils.loss import DiversityLoss

MIN_SAMPLES_PER_LABEL = 1


class Server(BaseServer):
    def __init__(self, generator, gen_epochs=1, gen_lr=1e-4,
                 ensemble_epoch=5, ensemble_lr=1e-4,
                 finetune_epochs=1,
                 lambda_cls=1., lambda_dis=1.,
                 distill_temperature=20.,  # temperature of distillation
                 **kwargs):
        super(Server, self).__init__(**kwargs)

        self.algorithm_name = "FedFTG"
        self.label_counts = None

        # finetune epochs in server
        self.finetune_epochs = finetune_epochs

        # config generator
        self.generator = generator
        self.generator.to(self.device)
        self.gen_epochs = gen_epochs
        gen_optim_kwargs = copy.deepcopy(self.optim_kwargs)
        gen_optim_kwargs['lr'] = gen_lr
        self.gen_optimizer = eval(self.opt_name)(self.generator.parameters(), **gen_optim_kwargs)

        # config ensemble
        self.ensemble_epoch = ensemble_epoch
        self.diversity_loss = DiversityLoss(metric='l1')

        cls_optim_kwargs = copy.deepcopy(self.optim_kwargs)
        cls_optim_kwargs['lr'] = ensemble_lr
        self.cls_optimizer = eval(self.opt_name)(self.model.classifier.parameters(), **gen_optim_kwargs)

        self.ensemble_weight = []

        # temperature of distillation
        self.distill_temperature = distill_temperature

        # Hyperparameter of updating generator, as in Eq.8,
        self.lambda_cls = lambda_cls
        self.lambda_dis = lambda_dis

    def aggregate(self):
        """aggregate clients' classifier"""
        # 1. Avg clients' model
        BaseServer.aggregate(self)

        # 2. estimate p and ensemble weight
        self.estimate_p()
        self.refresh_ensemble_weight()

        # 3. fine tune
        for e in range(self.finetune_epochs):
            # Sample batch of data. Here, our z is put in generator (line 4)
            batch_y = np.random.choice(range(self.num_classes),
                                       size=(self.batch_size,),
                                       p=self.label_counts / sum(self.label_counts))
            batch_y = torch.tensor(batch_y, dtype=torch.int64).to(self.device)

            self.model.classifier.to(self.device)
            for c in self.selected_clients:
                c.model.eval()
                c.model.to(self.device)

            # Update generator
            self.update_generator(batch_y)

            # Fine-tune model
            self.ensemble_fine_tune(batch_y)

    def estimate_p(self):
        # estimate sampling probability (line 2)
        self.label_counts = 0.
        for client in self.selected_clients:
            self.label_counts += np.array(client.label_counts)

    def refresh_ensemble_weight(self):
        # get refresh_ensemble_weight, i.e. alpha_ky as [client_k, class_y] (line 5)
        # Eq. 9
        self.ensemble_weight = []
        for client in self.selected_clients:
            temp = []
            for i, y_ki in enumerate(client.label_counts):
                temp.append(y_ki / self.label_counts[i])
            self.ensemble_weight.append(temp)

    def update_generator(self, y):
        self.generator.train()
        self.model.eval()

        for i in range(self.gen_epochs):
            # diversity loss
            z, eps = self.generator(y)
            dis_loss = self.diversity_loss(z, eps)

            # cls loss
            student_output = self.model.classifier(z)
            cls_loss = self.loss_fn(student_output, y)

            # md loss
            teacher_output = 0.
            for j, client in enumerate(self.selected_clients):
                client_output = client.model.classifier(z)
                weight = np.array(self.ensemble_weight[j])[y.to('cpu').numpy()].reshape(-1,1)
                expand_weight = np.tile(weight, (1, self.num_classes))
                expand_weight_tensor = torch.tensor(expand_weight, dtype=torch.float32).to(self.device)
                teacher_output += client_output * expand_weight_tensor
            md_loss = F.kl_div(F.softmax(student_output / self.distill_temperature, dim=-1),
                               F.softmax(teacher_output / self.distill_temperature, dim=-1),
                               reduction='batchmean')
            # Eq. 8
            loss = self.lambda_dis * dis_loss + self.lambda_cls * cls_loss - md_loss

            # backward & step optim
            self.gen_optimizer.zero_grad()
            loss.backward()
            self.gen_optimizer.step()

        self.generator.eval()

    def ensemble_fine_tune(self, y):
        self.model.classifier.train()
        self.generator.eval()

        for i in range(self.ensemble_epoch):
            z, eps = self.generator(y)
            student_output = self.model.classifier(z)
            cls_loss = self.loss_fn(student_output, y)

            teacher_output = 0.
            for j, client in enumerate(self.selected_clients):
                client_output = client.model.classifier(z)
                weight = np.array(self.ensemble_weight[j])[y.to('cpu').numpy()].reshape(-1,1)
                expand_weight = np.tile(weight, (1, self.num_classes))
                expand_weight_tensor = torch.tensor(expand_weight, dtype=torch.float32).to(self.device)
                teacher_output += client_output * expand_weight_tensor
            md_loss = F.kl_div(F.softmax(student_output / self.distill_temperature, dim=-1),
                               F.softmax(teacher_output / self.distill_temperature, dim=-1))

            loss = - self.lambda_cls * cls_loss + md_loss

            # backward & step optim
            self.cls_optimizer.zero_grad()
            loss.backward()
            self.cls_optimizer.step()

        self.model.classifier.eval()
