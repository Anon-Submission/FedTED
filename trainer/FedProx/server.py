from trainer.FedAvg.server import Server as Base_Server


class Server(Base_Server):
    def __init__(self, **config):
        super(Server, self).__init__(**config)
        self.algorithm_name = "FedProx"

    """Server working process of FedProx is same as FedAvg"""

