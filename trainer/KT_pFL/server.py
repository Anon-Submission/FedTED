import numpy as np
import torch
from trainer.FedMD.server import Server as BaseServer
import torch.nn.functional as F


class Server(BaseServer):
    def __init__(self, penalty_ratio=0.7, distill_temperature=20., **config):
        super(Server, self).__init__(**config)
        self.algorithm_name = "KT_pFL"

        # the knowledge coefficient matrix
        self.weight = torch.ones(size=(self.num_clients, self.num_clients)) / self.num_clients

        self.pho = penalty_ratio
        self.temperature = distill_temperature
        # self.raw_logits_matrix = None
        # self.weighted_logits_matrix = None
        # self.nk_vector = None

        self.clients_logits = None

    def distribute_model(self):
        """override the distribute_model func in FedMD, after receiving clients' logits, update c , cal new logits
        """
        # distribute new alignment_data
        self.alignment_data, _ = next(iter(self.alignment_loader))
        for client in self.selected_clients:
            client.alignment_data = self.alignment_data

        # clients cal logits
        zero_logits = self.selected_clients[0].get_logits() * 0.
        self.clients_logits = [self.clients[idx].get_logits()
                               if idx in self.selected_clients_ids else zero_logits
                               for idx in self.registered_client_ids]

        # update weighted logits matrix
        self.update_c()

        # use kt gat server logits
        client_logits_matrix = torch.stack(self.clients_logits, dim=-1)
        weighted_logits_matrix = self.weight_trans(self.weight, client_logits_matrix)
        # self.logits = weighted_logits_matrix.sum(dim=-1)

        # sent update logits to clients
        for idx in self.selected_clients_ids:
            self.clients[idx].glob_logits = weighted_logits_matrix[:, :, idx]

    def update_c(self):
        # clients' logits matrix before update c
        raw_logits_matrix = torch.stack(self.clients_logits, dim=-1)
        weighted_logits_matrix = self.weight_trans(self.weight, raw_logits_matrix)
        # weight_mean for mse
        weight_mean = torch.ones(size=(self.num_clients, self.num_clients)).float() / self.num_clients

        self.weight.requires_grad = True
        # === cal loss, note, In their Kt_pfl project code, the KLDivLoss is not used,
        # but reference to their paper, this need to be KLDivLoss for the first term.
        loss_value = F.kl_div(F.softmax(raw_logits_matrix / self.temperature, dim=-1),
                              F.softmax(weighted_logits_matrix / self.temperature, dim=-1),
                              reduction='batchmean') + F.mse_loss(self.weight, weight_mean)

        loss_value.backward()
        with torch.no_grad():
            lr = self._adaptive_lr(self.weight.grad, self.num_clients)
            self.weight.sub_(self.weight.grad * lr)
        self.weight.grad.zero_()
        self.weight.requires_grad = False

    @staticmethod
    def weight_trans(weight, logits_matrix):
        """weight transport the logits_matrix

        Args:
            weight: shape as [num_clients, num_clients], element means c_i->c_j weight
            logits_matrix: [num_data_alignment, num_classes, num_clients],
                meas the clients' logits for each data
        """
        num_clients = weight.size(0)
        assert num_clients == logits_matrix.size(-1), \
            f"weight size {num_clients}, logits long: {logits_matrix.size(-1)}"

        new_logits_list = []
        for i in range(num_clients):
            new_logits_i = torch.zeros_like(logits_matrix[:, :, i])
            for j in range(num_clients):
                new_logits_i += weight[i][j] * logits_matrix[:, :, j]
            new_logits_list.append(new_logits_i)

        new_logits_matrix = torch.stack(new_logits_list, dim=-1)
        return new_logits_matrix

    @staticmethod
    def _adaptive_lr(grad, num_clients):
        """adaptive lr_c for knowledge coefficient matrix
        same as github Kt_pfl project
        """
        grad_abs = torch.abs(grad)
        grad_sum = torch.sum(grad_abs)
        grad_avg = grad_sum.item() / num_clients
        lr = 1.0
        for i in range(5):
            if grad_avg > 0.01:
                grad_avg *= (1.0 / 5)
                lr /= 5
            if grad_avg < 0.01:
                grad_avg *= (1.0 * 5)
                lr *= 5
        return lr
