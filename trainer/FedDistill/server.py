from trainer.FedAvg.server import Server as BaseServer


class Server(BaseServer):
    """FedDistill Server
        num_alignment: size of public for distilling. Here we ignore this arg.
    """

    def __init__(self, fed_distill_aggregate=False, **config):
        """

        Args:
            fed_distill_aggregate: if aggregate model weights by avg. If False,
                vanilla FedDistill (weak but save communication resource), else, FedDistill + FedAvg.
        """
        super(Server, self).__init__(**config)
        self.algorithm_name = "FedDistill"
        self.fed_distill_aggregate = fed_distill_aggregate

    def distribute_model(self):
        """when distributing, use label logits"""
        # line 10-12
        bar_label_logits = [0. for _ in range(self.num_classes)]
        for client in self.selected_clients:
            for i in range(self.num_classes):
                bar_label_logits[i] += client.bar_label_logits[i]

        # line 13-15
        for client in self.selected_clients:
            for i in range(self.num_classes):
                client.hat_label_logits[i] = bar_label_logits[i] - client.bar_label_logits[i]
                client.hat_label_logits[i] /= (len(self.selected_clients) - 1)

    def aggregate(self):
        """do not aggregate model"""
        if self.fed_distill_aggregate:
            BaseServer.aggregate(self)
