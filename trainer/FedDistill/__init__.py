"""
FedDistill
The implementation of the paper:
    “Communication-Efficient On-Device Machine Learning:
    Federated Distillation and Augmentation under Non-IID Private Data,”
    in NeurIPS, 2018. Available: http://arxiv.org/abs/1811.11479.

    Since there is no offical code, we implement it ourselves.

    Besides, original FedDistill is for lower communication, which makes it too weak when do not share model weights
    Therefore, we do same as Kt_pFL, FedMD, i.e. plus FedDistill and FedAvg.
"""