from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .trainer import Trainer
from .train_classification import ClassificationTrainer
from .uncertainty_bnn import UncertaintyTrainer
from .train_regression import LinearRegTrainer
from .r2d2_bnn_linreg import R2D2LinearRegTrainer
from .horseshoe_bnn_linreg import HorseshoeLinearRegTrainer
from .train_mcdropout_linreg import MCDLinearRegTrainer

__all__ = [
    'Trainer',
]
