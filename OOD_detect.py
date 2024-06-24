import argparse
import logging
import sys
sys.path.append("UE")
import warnings

import numpy as np
import pandas as pd

import os
import torch
import torch.nn as nn
# from sklearn.metrics import (roc_auc_score, roc_curve,accuracy_score,classification_report,confusion_matrix)
from BNN.models import ABMIL, BClassifier
import wandb
from pyhealth.metrics import binary_metrics_fn, multiclass_metrics_fn
from dataset import BagDataset
from torch.utils.data import DataLoader
# from Opt.lookahead import Lookahead
# from Opt.radam import RAdam
# from BNN.models.DTFD.network import DimReduction, get_cam_1d
# from BNN.models.DTFD.Attention import Attention_Gated as Attention
# from BNN.models.DTFD.Attention import Attention_with_Classifier, Classifier_1fc
import random
import time
import copy
def get_UM(data_loader, args):
    if args.model=='abmil':
        milnet = BClassifier(args.feats_size)
    milnet.load_state_dict(torch.load(args.model_dir))
    milnet.eval()
    test_slides = []
    correct_predictions = []
    max_probs = []
    entropies = []
    with torch.no_grad():
        for i, (bag_label, bag_feats, slide) in enumerate(test_loader):
            bag_label = bag_label.cuda()
            bag_feats = bag_feats.cuda()
            bag_feats = bag_feats.view(-1, args.feats_size)
            bag_prediction = milnet(bag_feats)
            slide = slide[0].split('\n')[0]
            if args.num_classes == 1:
                bag_prediction = torch.sigmoid(bag_prediction).squeeze().cpu().numpy().item()
                probs = np.array([1 - bag_prediction, bag_prediction])
                max_prob = np.max(probs)
                entropy = -np.sum(probs * np.log(probs))
                pred = np.round(bag_prediction)
            else:
                bag_prediction = torch.nn.Softmax(dim=1)(bag_prediction)
                conf = torch.max(bag_prediction).item()
            if pred != bag_label:
                correct_predictions.append(1)
            else:
                correct_predictions.append(0)

            max_probs.append(max_prob)
            entropies.append(entropy)
            test_slides.append(slide)

    df = pd.DataFrame(
        {'slide': test_slides, 'correct_predictions': correct_predictions, 'max_prob': max_probs, 'entropy': entropies})
    df.to_csv(os.path.join(args.save_path, args.model, 'UM.csv'), index=False)
def main():
    parser = argparse.ArgumentParser(description='OOD')
    parser.add_argument('--extractor', type=str, default='Kimia', help='extractor name')
    # parser.add_argument('--task', type=str, default='binary', help='task name')
    parser.add_argument('--dataset', type=str, default='Camelyon', help='dataset name')
    parser.add_argument('--out_dataset', type=str, default='COAD', help='dataset name')
    parser.add_argument('--model_dir', type=str, default=None, help='dir to the saved model')
    # parser.add_argument('--lr', default=0.0002, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--save_path', type=str, default='Weights', help='dir to save models')
    parser.add_argument('--model', type=str, default='abmil', help='model name')
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers')
    parser.add_argument('--feats_size', type=int, default=1024, help='feature size')
    # parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay [5e-3]')
    # parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs')
    # parser.add_argument('--early_stop', type=int, default=20, help='early stop')
    # parser.add_argument('--rep',type=int,default=9)
    parser.add_argument('--res_dir', type=str, default=None, help='dir to save results')
    args = parser.parse_args()
    dataset_path = os.path.join('datasets_csv', args.dataset,
                              f'OOD_{args.dataset}_{args.out_dataset}' + '.csv')
    dataset = BagDataset(train_path, args)
    data_loader = DataLoader(dataset, 1, shuffle=True, num_workers=args.num_workers)
    if args.res_dir == None:
        get_UM(data_loader, args)

