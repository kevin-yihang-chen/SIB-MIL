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
from sklearn.metrics import (roc_auc_score, roc_curve,accuracy_score,classification_report,confusion_matrix)
from BNN.models import ABMIL, BClassifier
import wandb
from pyhealth.metrics import binary_metrics_fn, multiclass_metrics_fn
from dataset import BagDataset
from torch.utils.data import DataLoader
from Opt.lookahead import Lookahead
from Opt.radam import RAdam
from BNN.models.DTFD.network import DimReduction, get_cam_1d
from BNN.models.DTFD.Attention import Attention_Gated as Attention
from BNN.models.DTFD.Attention import Attention_with_Classifier, Classifier_1fc
import random
import time
import copy
warnings.simplefilter('ignore')

def train(train_df, milnet, criterion, optimizer, args, n_train, weight_kl):
    milnet.train()
    total_loss = 0
    for i, (bag_label, bag_feats, _) in enumerate(train_df):
        optimizer.zero_grad()
        if torch.isnan(bag_feats).sum() > 0:
            continue
        bag_label = bag_label.cuda()
        bag_feats = bag_feats.cuda()
        bag_feats = bag_feats.view(-1, args.feats_size)
        if args.model == 'abmil':
            bag_prediction = milnet(bag_feats)
            bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            loss = bag_loss
        elif args.model == 'abuamil':
            bag_prediction = milnet(bag_feats, Train_flag=True, train_sample=n_train)
            bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            loss = bag_loss
            loss = loss + milnet.kl_loss() * weight_kl
        else:
            raise NotImplementedError
        loss.backward()
        optimizer.step()
        if args.model == 'abuamil':
            milnet.analytic_update()
        total_loss = total_loss + loss.item()
        sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f' % (i, len(train_df), loss.item()))
    sys.stdout.write('\n')
    return total_loss / len(train_df)


def test(test_df, milnet, criterion, args, n_test):
    milnet.eval()
    total_loss = 0
    test_labels = []
    test_predictions = []
    with torch.no_grad():
        for i, (bag_label, bag_feats, _) in enumerate(test_df):
            # bag_label, bag_feats = get_bag_feats_v2(test_feats[i], test_gts[i], args)
            bag_label = bag_label.cuda()
            bag_feats = bag_feats.cuda()
            bag_feats = bag_feats.view(-1, args.feats_size)
            if args.model == 'abmil':
                bag_prediction = milnet(bag_feats)
                bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
                loss = bag_loss
            elif args.model == 'abuamil':
                bag_prediction = milnet(bag_feats, Train_flag=False, test_sample=n_test)
                bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
                loss = bag_loss
            else:
                raise NotImplementedError
            total_loss = total_loss + loss.item()
            sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (i, len(test_df), loss.item()))
            test_labels.extend([bag_label.cpu().numpy()])
            if args.num_classes == 1:
                test_predictions.extend([(torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])
            else:
                test_predictions.extend([(torch.nn.Softmax(dim=1)(bag_prediction)).squeeze().cpu().numpy()])
        sys.stdout.write('\n')
    test_labels = np.array(test_labels)
    test_labels = test_labels.squeeze()
    # test_labels = test_labels.reshape(len(test_labels), -1)
    test_predictions = np.array(test_predictions)
    # y_pred, y_true = inverse_convert_label(test_predictions), inverse_convert_label(test_labels)
    if args.num_classes == 1:
        res = binary_metrics_fn(test_labels, test_predictions,
                                metrics=['accuracy', 'precision', 'recall', 'roc_auc', 'f1'])
        acc = res['accuracy']
        p = res['precision']
        r = res['recall']
        f1 = res['f1']
        c_auc = res['roc_auc']
        avg = np.mean([p, r, acc, f1])
        return p, r, acc, f1, avg, c_auc
    else:
        res = multiclass_metrics_fn(np.where(test_labels == 1)[1], test_predictions,
                                    metrics=["roc_auc_weighted_ovo", "f1_weighted", "accuracy"])
        acc = res['accuracy']
        f = res['f1_weighted']
        # r = res['recall']
        c_auc = res['roc_auc_weighted_ovo']
        return acc, f, c_auc

def generate_UM(args, test_loader, milnet):
    if args.model == 'abmil':
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
                    probs = np.array([1-bag_prediction, bag_prediction])
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

        df = pd.DataFrame({'slide': test_slides, 'correct_predictions': correct_predictions, 'max_prob': max_probs,'entropy':entropies})
        df.to_csv(os.path.join(args.save_path, args.model, 'UM.csv'), index=False)

def main():
    parser = argparse.ArgumentParser(description='UM')
    parser.add_argument('--extractor', type=str, default='Kimia', help='extractor name')
    parser.add_argument('--task', type=str, default='binary', help='task name')
    parser.add_argument('--dataset', type=str, default='Camelyon', help='dataset name')
    parser.add_argument('--model_dir', type=str, default=None, help='dir to the saved model')
    parser.add_argument('--lr', default=0.0002, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--save_path', type=str, default='Weights', help='dir to save models')
    parser.add_argument('--model', type=str, default='abmil', help='model name')
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers')
    parser.add_argument('--feats_size', type=int, default=1024, help='feature size')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--early_stop', type=int, default=20, help='early stop')
    parser.add_argument('--rep',type=int,default=9)
    args = parser.parse_args()
    n_train = 1
    n_test = 1
    kl = 1e-6
    early_stop = args.early_stop

    print(f'current args: {args}')

    if args.task == 'binary':
        args.num_classes = 1
    elif args.task == 'staging':
        args.num_classes = {'COAD': 4, 'BRACS_WSI': 3, 'BRCA': 4}[args.dataset]

    prior = {'horseshoe_scale': None, 'global_cauchy_scale': 1., 'weight_cauchy_scale': 1.,
             'beta_rho_scale': -5.,
             'log_tau_mean': None, 'log_tau_rho_scale': -5., 'bias_rho_scale': -5., 'log_v_mean': None,
             'log_v_rho_scale': -5.}

    if args.model == 'abmil':
        milnet = BClassifier(args.feats_size, args.num_classes).cuda()
    elif args.model == 'abuamil':
        milnet = ABMIL(args.feats_size, args.num_classes, layer_type='HS', priors=prior,
                       activation_type='relu').cuda()
    else:
        raise NotImplementedError
    train_path = os.path.join('datasets_csv', args.dataset,
                              f'{args.task}_{args.dataset}_train' + '.csv')
    train_path = pd.read_csv(train_path)
    test_path = os.path.join('datasets_csv', args.dataset,
                             f'{args.task}_{args.dataset}_testval' + '.csv')
    test_path = pd.read_csv(test_path)

    trainset = BagDataset(train_path, args)
    train_loader = DataLoader(trainset, 1, shuffle=True, num_workers=args.num_workers)
    testset_in = BagDataset(test_path, args)
    test_loader_in = DataLoader(testset_in, 1, shuffle=True, num_workers=args.num_workers)


    if args.model_dir == None:
        for i in range(args.rep):
            if args.num_classes == 1:
                criterion = nn.BCEWithLogitsLoss()
            else:
                criterion = nn.CrossEntropyLoss()

            optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, betas=(0.5, 0.9),
                                         weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)


            best_acc = 0
            count = 0
            os.makedirs(os.path.join(args.save_path, args.model), exist_ok=True)
            for epoch in range(args.num_epochs):
                if count >= early_stop:
                    break
                print(f'Early Stopping [{count}/{early_stop}]...')
                train_loss_bag = train(train_loader, milnet, criterion, optimizer, args, n_train, weight_kl=kl)
                if args.num_classes == 1:
                    precision, recall, accuracy, f1, avg, auc = test(test_loader, milnet, criterion, args,
                                                                     n_test)
                    print(f'pre:{precision},recall:{recall},acc:{accuracy},f1:{f1},auc:{auc}.')

                else:
                    accuracy, f1, auc = test(test_loader, milnet, criterion, args, n_test)
                    print(f'acc:{accuracy},f1:{f1},auc:{auc}.')

                logging.info('Epoch [%d/%d] train loss: %.4f' % (epoch, args.num_epochs, train_loss_bag))
                if args.model != 'transmil':
                    scheduler.step()
                if accuracy > best_acc:
                    print('saving model...')
                    best_acc = accuracy
                    torch.save(milnet.state_dict(), os.path.join(args.save_path, args.model, f'model_{i+1}.pth'))
                    count = 0
                else:
                    count += 1
    else:
        print('loading model...')
        milnet.load_state_dict(torch.load(args.model_dir))
        generate_UM(args, test_loader, milnet)



if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
