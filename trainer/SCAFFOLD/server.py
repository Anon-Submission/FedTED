import copy

from trainer.FedAvg.server import Server as Base_Server


class Server(Base_Server):
    def __init__(self, **config):
        super(Server, self).__init__(**config)
        self.algorithm_name = "SCAFFOLD"

        self.model.to(self.device)
        # init c in server
        self.control_global = copy.deepcopy(self.model.state_dict())

        # init all clients' control_local with same weight as servers' control_global
        for c in self.clients:
            c.control_local = copy.deepcopy(self.model.state_dict())

        # init delta_c
        self.delta_c = copy.deepcopy(self.model.state_dict())
        self.delta_x = copy.deepcopy(self.model.state_dict())

        self.model.to('cpu')

    def distribute_model(self):
        """distribute model and controls"""
        w = self.model.state_dict()
        for client in self.selected_clients:
            client.model.load_state_dict(w)
            client.control_global = self.control_global

    def aggregate(self):
        """aggregate update grads (line 16-17 in paper)"""
        # init delta_c, delta_x as 0
        for k in self.delta_c:
            self.delta_c[k] = 0.
        for k in self.delta_x:
            self.delta_x[k] = 0.

        # 1. calculate delta_x and delta_c: line 16
        m = len(self.selected_clients)
        for c in self.selected_clients:
            if self.glob_iter == 0:
                client_weight = c.model.state_dict()
                for k in self.delta_c:
                    self.delta_x[k] += client_weight[k].float()
            else:
                for k in self.delta_c:
                    self.delta_x[k] += c.delta_y[k].float()
                    self.delta_c[k] += c.delta_c[k].float()
        for k in self.delta_c:
            self.delta_x[k] /= m
            self.delta_c[k] /= m

        # 2. update global control variate: line 17
        self.model.to(self.device)
        global_weights = self.model.state_dict()
        for k in self.control_global:
            if self.glob_iter == 0:
                global_weights[k] = self.delta_x[k]
            else:
                global_weights[k] = global_weights[k].float() + self.delta_x[k]
                self.control_global[k] = self.control_global[k].float() + (m / self.num_clients) * self.delta_c[k]

        self.model.load_state_dict(global_weights)
