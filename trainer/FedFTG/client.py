from ..FedAvg.client import Client as BaseClient
import torch


class Client(BaseClient):
    def __init__(self, **kwargs):
        super(Client, self).__init__(**kwargs)
        # count label of client
        self.label_counts = [0 for _ in range(self.num_classes)]
        self.init_label_counts()

    def init_label_counts(self):
        # count num of samples for each label in the dataset
        for x, y in self.train_loader:
            for i in range(self.num_classes):
                idx = torch.nonzero(y == i).view(-1)
                self.label_counts[i] += len(idx)
