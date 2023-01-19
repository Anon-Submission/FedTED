import copy
import torch
from trainer.FedAvg.client import Client as BaseClient
from collections import OrderedDict

class Client(BaseClient):
    """
    Client for SCAFFOLD
        mode: the mode of update control variate in line 12 of paper pseudocode
    """

    def __init__(self, **config):
        super(Client, self).__init__(**config)

        # init c, c_i
        self.control_global = None  # assign when selected

        # state_dict of models, init by server
        self.control_local = None

        # delta c, delta y; assign after local update
        self.delta_y = OrderedDict()
        self.delta_c = OrderedDict()

    """In SCAFFOLD, the local update not only change model weights by loss,
             but also control variables. Thus, we rewrite one here."""

    def update(self, epochs=1, verbose=0):
        assert epochs > 0

        # step 1. init control w
        self.model.to(self.device)
        self.model.train()

        global_weights = self.model.state_dict()

        # step 2. local update (line 8-11 in paper)
        count = 0
        for epoch in range(epochs):
            # init loss value
            loss_value, num_samples = 0, 0
            # one epoch train
            for i, (x, y) in enumerate(self.train_loader):
                # put tensor into same device
                x, y = x.to(self.device), y.to(self.device)
                # calculate loss
                y_ = self.model(x)
                loss = self.loss_fn(y_, y)
                # backward & step optim
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # line 10 in alg, update model by sum weights and control variate
                local_weights = self.model.state_dict()
                for k in local_weights:
                    local_weights[k] = local_weights[k] - \
                                       self.lr * (self.control_global[k] - self.control_local[k])
                # update local model params
                self.model.load_state_dict(local_weights)
                count += 1

                # get loss valur of current bath
                loss_value += loss.item()
                num_samples += y.size(0)

            loss_value = loss_value / num_samples

        # step 3. update local control variate (line 12-14 in paper)
        # init c+, delta_ci, delta_y_i
        new_control_local = copy.deepcopy(self.model.state_dict())
        model_weights = self.model.state_dict()

        # update c, as discussed in the paper
        for k in model_weights:
            # line 12 in algo
            new_control_local[k] = new_control_local[k] - self.control_global[k]\
                                   + (global_weights[k] - model_weights[k]) / (count * self.lr)
            # return gradient y and c in line 13
            self.delta_c[k] = new_control_local[k] - self.control_local[k]
            self.delta_y[k] = model_weights[k] - global_weights[k]

        # update control_local
        if self.glob_iter != 0:  # not given in paper, but in code
            self.control_local = new_control_local

        # step 4. release gpu resource
        self.model.to('cpu')
        torch.cuda.empty_cache()
