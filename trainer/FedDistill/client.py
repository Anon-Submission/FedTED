import torch
from trainer.FedAvg.client import Client as BaseClient


class Client(BaseClient):
    """ FedDistill client """

    def __init__(self, fed_distill_gamma=1e-3, early_exit: int = None, **config):
        """

        Args:
            gamma: sum weight of distill loss and empirical loss. This value is not give by original paper.
                But according to our test, it's value should not be larger than 1e-4
        """
        super(Client, self).__init__(**config)

        self.bar_label_logits = [0. for _ in range(self.num_classes)]
        self.hat_label_logits = [0. for _ in range(self.num_classes)]
        self.cnt = [0] * self.num_classes

        self.init_bar_label_logits()

        self.gamma = fed_distill_gamma
        self.early_exit = early_exit if early_exit is not None \
            else self.total_rounds

    def update(self, epochs=1, verbose=0):
        # step 1. local update
        # a. model init
        self.model.to(self.device)
        self.model.train()
        label_logits = [0.  # torch.zeros(size=(self.num_classes,), device=self.device)
                        for _ in range(self.num_classes)]

        # b. train loop, line 3-6
        loss_metric = []  # to record avg loss
        for epoch in range(epochs):
            # init loss value
            loss_value, num_samples = 0, 0
            # one epoch train
            for i, (x, y) in enumerate(self.train_loader):
                # put tensor into same device
                x, y = x.to(self.device), y.to(self.device)
                # calculate loss, line 5
                y_ = self.model(x)
                loss = self.loss_fn(y_, y)
                if self.glob_iter < 10:  # if all use, will collapse
                    loss += self.gamma * self.distill_loss_fn(y_, y)

                # backward & step optim
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # update lable logits and label counts, line 6
                self.model.eval()
                with torch.no_grad():
                    label_logits = self.add_label_logits(label_logits, self.model(x), y)
                self.model.train()
                # get loss valur of current bath
                loss_value += loss.item()
                num_samples += y.size(0)

            # Use mean loss value of each epoch as metric
            # Just a reference value, not precise. If you want precise, dataloader should set `drop_last = True`.
            loss_value = loss_value / num_samples
            loss_metric.append(loss_value)

        # step 2. update hat label logits line 7-8
        for i in range(self.num_classes):
            if self.cnt[i] > 0:
                self.bar_label_logits[i] = label_logits[i] / self.cnt[i]

        # step 3. release gpu resource
        self.model.to('cpu')
        torch.cuda.empty_cache()

        avg_loss = sum(loss_metric) / len(loss_metric)
        return avg_loss

    def init_bar_label_logits(self):
        self.model.to(self.device)
        self.model.eval()
        label_logits = [0. for _ in range(self.num_classes)]
        for i, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)
            with torch.no_grad():
                label_logits = self.add_label_logits(label_logits, self.model(x), y)
            self.count_labels(y)
        for i in range(self.num_classes):
            if self.cnt[i] > 0:
                self.bar_label_logits[i] = label_logits[i] / self.cnt[i]

    def add_label_logits(self, label_logits, y_, y):
        for i in range(self.num_classes):
            idx = torch.nonzero(y == i).view(-1)
            if len(idx) > 0:
                label_logits[i] += y_[idx].sum(dim=0)
        return label_logits

    def count_labels(self, y):
        """count number of labels in a batch."""
        for i in range(self.num_classes):
            idx = torch.nonzero(y == i).view(-1)
            self.cnt[i] += len(idx)

    def distill_loss_fn(self, output, labels):
        # new loss fn according to line 5
        batch_hat_label_logits = torch.stack(
            [self.hat_label_logits[labels[i]] for i in range(labels.size(0))],
            dim=0)
        loss_value = self.loss_fn(output, batch_hat_label_logits)
        return loss_value

    def adaptive_gamma(self):
        self.gamma = self.gamma * (0.5 ** self.glob_iter)
