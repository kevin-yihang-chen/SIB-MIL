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
import sklearn
from sklearn.metrics import (roc_auc_score, roc_curve, accuracy_score,
                             classification_report, confusion_matrix, precision_recall_curve)
from BNN.models import ABMIL, BClassifier, BClassifier_Dropout
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
from losses import EdlLoss
import torch.distributions as dist
from torch.distributions.dirichlet import Dirichlet
warnings.simplefilter('ignore')

def one_hot_embedding(labels, num_classes=10):
    # Convert to One Hot Encoding
    y = torch.eye(num_classes).cuda()
    labels = labels[0].type(torch.int64)
    return y[labels]


def target_alpha(targets, num_classes):
    target = targets.cpu().numpy()

    def gen_onehot(category, num_classes):
        label = np.ones(num_classes)
        label[int(category)] = 10
        return label

    target_alphas = []
    for i in target:
        if i == 200:
            target_alphas.append(np.ones(200))
        else:
            target_alphas.append(gen_onehot(i, num_classes))
    return torch.Tensor(target_alphas)

def train(train_df, milnet, criterion, optimizer, args, n_train, weight_kl, epoch):
    if args.model == 'abmil_ensemble':
        for j in range(args.ensemble_size):
            milnet[j].train()
    else:
        milnet.train()
    total_loss = 0
    for i, (bag_label, bag_feats, _) in enumerate(train_df):
        optimizer.zero_grad()
        if torch.isnan(bag_feats).sum() > 0:
            continue
        bag_label = bag_label.cuda()
        bag_feats = bag_feats.cuda()
        bag_feats = bag_feats.view(-1, args.feats_size)
        if args.model == 'abmil' or args.model == 'abmil_dropout':
            bag_prediction = milnet(bag_feats)
            bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            loss = bag_loss
        elif args.model == 'abuamil':
            bag_prediction = milnet(bag_feats, Train_flag=True, train_sample=n_train)
            bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            loss = bag_loss
            loss = loss + milnet.kl_loss() * weight_kl
        elif args.model == 'abmil_edl':
            bag_prediction = milnet(bag_feats)
            bag_label = one_hot_embedding(bag_label, args.num_classes).cuda()
            bag_loss = criterion.edl_digamma_loss(bag_prediction, bag_label, epoch, args.num_classes, 10)
            loss = bag_loss
        elif args.model == 'abmil_dpn':
            bag_prediction = milnet(bag_feats)
            alpha = target_alpha(bag_label, args.num_classes).cuda()
            prob = nn.Softmax(dim=1)(bag_prediction)
            output_alpha = torch.exp(prob)
            dirichlet1 = Dirichlet(output_alpha)
            dirichlet2 = Dirichlet(alpha)
            loss = torch.sum(dist.kl.kl_divergence(dirichlet1, dirichlet2))
        elif args.model == 'abmil_ensemble':
            bag_prediction = [milnet[j](bag_feats) for j in range(args.ensemble_size)]
            bag_prediction = torch.stack(bag_prediction, dim=0)
            bag_prediction = bag_prediction.mean(dim=0)
            bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            loss = bag_loss
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

def test_ood_detection(test_loader, model, args, n_test):
    test_scores = []
    labels = []
    if args.model == 'abmil_dropout':
        model.train()
    elif args.model == 'abmil_ensemble':
        [model[j].eval() for j in range(args.ensemble_size)]
    else:
        model.eval()
    for i, (bag_label, bag_feats, _) in enumerate(test_loader):
        bag_label = bag_label.cuda()
        bag_feats = bag_feats.cuda()
        bag_feats = bag_feats.view(-1, args.feats_size)
        with torch.no_grad():
            if args.model in ['abmil','abmil_dropout','abmil_edl','abmil_dpn']:
                bag_prediction = model(bag_feats)
            elif args.model == 'abuamil':
                bag_prediction = model(bag_feats, Train_flag=False, test_sample=n_test)
            elif args.model == 'abmil_ensemble':
                bag_prediction = [model[j](bag_feats) for j in range(args.ensemble_size)]
                bag_prediction = torch.stack(bag_prediction, dim=0)
                bag_prediction = bag_prediction.mean(dim=0)
            if args.num_classes == 1:
                bag_prediction = torch.sigmoid(bag_prediction).squeeze().cpu().numpy().item()
            else:
                bag_prediction = torch.nn.Softmax(dim=1)(bag_prediction).squeeze().cpu().numpy()
        entropy = - np.log(bag_prediction) * bag_prediction
        test_scores.append(np.sum(entropy))
        labels.append(bag_label.cpu().numpy())

    def format_scores(scores):

        scores[np.isposinf(scores)] = 1e9
        maximum = np.amax(scores)
        scores[np.isposinf(scores)] = maximum + 1


        scores[np.isneginf(scores)] = -1e9
        minimum = np.amin(scores)
        scores[np.isneginf(scores)] = minimum - 1

        scores[np.isnan(scores)] = 0

        return scores

    scores = format_scores(np.array(test_scores))
    labels_1 = np.array(labels)
    labels_2 = 1-labels_1
    def comp_aucs_ood(scores, labels_1, labels_2):
        labels_1 = labels_1.flatten()
        labels_2 = labels_2.flatten()
        auroc_1 = roc_auc_score(labels_1, scores)
        auroc_2 = roc_auc_score(labels_2, scores)
        auroc = max(auroc_1, auroc_2)

        precision, recall, thresholds = precision_recall_curve(labels_1, scores)
        aupr_1 = sklearn.metrics.auc(recall, precision)

        precision, recall, thresholds = precision_recall_curve(labels_2, scores)
        aupr_2 = sklearn.metrics.auc(recall, precision)

        aupr = max(aupr_1, aupr_2)

        return auroc, aupr, precision, recall

    auroc, aupr, precision, recall = comp_aucs_ood(scores, labels_1, labels_2)

    return auroc, aupr, precision, recall

def main():
    parser = argparse.ArgumentParser(description='UM')
    parser.add_argument('--extractor', type=str, default='Kimia', help='extractor name')
    parser.add_argument('--task', type=str, default='binary', help='task name')
    parser.add_argument('--dataset', type=str, default='Camelyon', help='dataset name')
    parser.add_argument('--dataset_out', type=str, default='COAD', help='OOD dataset name')
    parser.add_argument('--model_dir', type=str, default=None, help='dir to the saved model')
    parser.add_argument('--lr', default=0.0002, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--save_path', type=str, default='Weights', help='dir to save models')
    parser.add_argument('--model', type=str, default='abmil', help='model name')
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers')
    parser.add_argument('--feats_size', type=int, default=1024, help='feature size')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--rep',type=int,default=9)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--ensemble_size', type=int, default=5)
    args = parser.parse_args()
    if args.model != 'abmil_ensemble':
        args.ensemble_size = None
    n_train = 1
    n_test = 1
    kl = 1e-6
    for i in range(args.rep):
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
        elif args.model == 'abmil_dropout':
            milnet = BClassifier_Dropout(args.feats_size, args.num_classes).cuda()
        elif args.model == 'abuamil':
            milnet = ABMIL(args.feats_size, args.num_classes, layer_type='HS', priors=prior,
                           activation_type='relu').cuda()
        elif args.model == 'abmil_edl' or args.model == 'abmil_dpn':
            args.num_classes = 2
            milnet = BClassifier(args.feats_size, args.num_classes).cuda()
        elif args.model == 'abmil_ensemble':
            milnet = [BClassifier(args.feats_size, args.num_classes).cuda() for _ in range(args.ensemble_size)]
        else:
            raise NotImplementedError
        train_path = os.path.join('datasets_csv', args.dataset,
                                  f'{args.task}_{args.dataset}_train' + '.csv')
        train_path = pd.read_csv(train_path)
        test_path = os.path.join('datasets_csv', args.dataset,
                                 f'OOD_{args.dataset}_{args.dataset_out}' + '.csv')
        test_path = pd.read_csv(test_path)
        trainset = BagDataset(train_path, args)
        train_loader = DataLoader(trainset, 1, shuffle=True, num_workers=args.num_workers)
        testset = BagDataset(test_path, args)
        test_loader = DataLoader(testset, 1, shuffle=True, num_workers=args.num_workers)
        if args.wandb:
            wandb.init(name=f'OOD_{args.model}',
                       project='UAMIL_OOD',
                       entity='yihangc',
                       notes='',
                       mode='online',  # disabled/online/offline
                       config=args,
                       tags=[])


        if args.model_dir == None:
            best_auc = 0
            if args.num_classes == 1:
                criterion = nn.BCEWithLogitsLoss()
            else:
                criterion = nn.CrossEntropyLoss()
            if args.model == 'abmil_edl':
                criterion = EdlLoss(device=0)
            # if args.model == 'bayesmil':
            #     criterion = nn.CrossEntropyLoss()
            #     print('\nInit Model...', end=' ')
            #     model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
            #     model_dict.update({"size_arg": args.model_size})
            #     model = bMIL_model_dict[args.model_type.split('-')[1]](**model_dict)
            #     bayes_args = [get_ard_reg_vdo, 1e-8, 1e-6]
            #     bayes_args.append('vis')
            if args.model == 'abmil_ensemble':
                optimizer = torch.optim.Adam([{'params': milnet[j].parameters()} for j in range(args.ensemble_size)],
                                             lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
            else:
                optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, betas=(0.5, 0.9),
                                             weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)
            os.makedirs(os.path.join(args.save_path, args.model), exist_ok=True)
            for epoch in range(args.num_epochs):
                print(f'Epoch [{epoch + 1}/{args.num_epochs}]...')
                train(train_loader, milnet, criterion, optimizer, args, n_train, weight_kl=kl, epoch=epoch)
                if args.model != 'transmil':
                    scheduler.step()
                auroc, aupr, precision, recall = test_ood_detection(test_loader, milnet, args, n_test)
                if args.wandb:
                    wandb.log({'auroc': auroc, 'aupr': aupr})
                print(f'auroc:{auroc},aupr:{aupr}.')
                if auroc > best_auc:
                    print('saving model...')
                    best_auc = auroc
                    if args.model == 'abmil_ensemble':
                        for j in range(args.ensemble_size):
                            torch.save(milnet[j].state_dict(), os.path.join(args.save_path, args.model, f'OOD_model_{i}_{j}.pth'))
                    else:
                        torch.save(milnet.state_dict(), os.path.join(args.save_path, args.model, f'OOD_model_{i}.pth'))
            wandb.finish()
        # else:
        #     print('loading model...')
        #     milnet.load_state_dict(torch.load(args.model_dir))
        #     generate_UM(args, test_loader, milnet)



if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
