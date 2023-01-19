"""
FedProx
The implementation of the paper:
    Li T, Sahu AK, Zaheer M, Sanjabi M, Talwalkar A, Smith V.
    Federated optimization in heterogeneous networks. In: MLSys 2020:5132–5143.
Modified with reference to projects:
    https://github.com/litian96/FedProx
    https://github.com/ongzh/ScaffoldFL
"""

from .client import Client
from .server import Server
