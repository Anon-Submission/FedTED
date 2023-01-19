from torch.utils.data import Dataset, DataLoader
from torch.optim import *

from trainer.FedAvg.server import Server as BaseServer
import copy
from utils.base_train import distill_by_models
from utils.loss import VanillaKDLoss

class Server(BaseServer):
    def __init__(self, public_dataset: Dataset, ensemble_epoch=5, ensemble_lr=0.0001,
                 distill_temperature=20., **config):
        super(Server, self).__init__(**config)
        self.algorithm_name = "FedDF"

        self.public_dataset = public_dataset
        self.distill_loader = DataLoader(self.public_dataset, batch_size=self.batch_size, shuffle=True)

        self.ensemble_epoch = ensemble_epoch

        ensemble_optim_kwargs = copy.deepcopy(self.optim_kwargs)
        ensemble_optim_kwargs['lr'] = ensemble_lr
        self.ensemble_optimizer = eval(self.opt_name)(self.model.parameters(), **ensemble_optim_kwargs)

        self.ensemble_loss_fn = VanillaKDLoss(temperature=distill_temperature)

        self.temperature = distill_temperature

    def aggregate(self):
        """The FedDF use ensemble distillation after FedAvg aggregate"""
        # 1. avg aggregate
        BaseServer.aggregate(self=self)

        # 2. ensemble distillation
        teacher_models = [client.model for client in self.selected_clients]
        distill_by_models(self.model, teacher_models, self.distill_loader, self.ensemble_optimizer,
                          self.ensemble_loss_fn, self.ensemble_epoch, self.device,)
