from trainer.FedAvg.server import Server as Base_Server
from utils.nets import TwinBranchNets

class Server(Base_Server):
    def __init__(self, **kwargs):
        super(Server, self).__init__(**kwargs)

        assert isinstance(self.model, TwinBranchNets), \
            "FedRod need model in format of [feature_extractor, classifier]. Now, only TwinBranchNets is ok."
        self.algorithm_name = "FedRoD"

    """Server working process of FedRoD is same as FedAvg, 
    but the distribute and aggregate only do for generic head"""

    def distribute_model(self):
        feature_extractor_w = self.model.feature_extractor.state_dict()
        classifier_w = self.model.classifier.state_dict()
        # clients' classifier_p won't be shared anyway.
        for client in self.selected_clients:
            client.feature_extractor.load_state_dict(feature_extractor_w)
            client.classifier_g.load_state_dict(classifier_w)

    def aggregate(self):
        """
        aggregate the updated and transmitted parameters from each selected client.
        """
        # aggregate feature_extractor
        msg_list = [(client.num_samples, client.feature_extractor.state_dict())
                    for client in self.selected_clients]
        w_dict = self.avg_weights(msg_list)

        self.model.feature_extractor.load_state_dict(w_dict)

        # aggregate classifier
        msg_list = [(client.num_samples, client.classifier_g.state_dict())
                    for client in self.selected_clients]
        w_dict = self.avg_weights(msg_list)

        self.model.classifier.load_state_dict(w_dict)
