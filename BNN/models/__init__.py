from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from .abmil import ABMIL, BClassifier, BClassifier_Dropout
from .dsmil import DSMIL
from .TransMIL import TransMIL

__all__ = [
    "CNN",
    "LeNet",
    "AlexNet",
    "SimpleCNN",
    "ResNet"
]
