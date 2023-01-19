from trainer.FedAvg.client import Client as BaseClient


class Client(BaseClient):
    def __init__(self, **config):
        super(Client, self).__init__(**config)
