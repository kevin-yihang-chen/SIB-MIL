# This script is modfied from https://github.com/binli123/dsmil-wsi/blob/master/train_tcga.py

import argparse
import logging
import sys
sys.path.append(".")
import warnings

import numpy as np
import pandas as pd

import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import torch
import torch.nn as nn
from sklearn.metrics import (roc_auc_score, roc_curve,accuracy_score,classification_report,confusion_matrix)
from BNN.models import ABMIL, DSMIL, TransMIL

# from model import abmil, dsmil
# from tools.utils import setup_logger
# from model.dpmil import DirichletProcess
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


def convert_label(labels, num_classes=2):
    # one-hot encoding for multi-class labels
    if num_classes > 1:
        # one-hot encoding
        converted_labels = np.zeros((len(labels), num_classes))
        for ix in range(len(labels)):
            converted_labels[ix, int(labels[ix])] = 1
        return converted_labels
    else:
        # return binary labels
        return labels


def inverse_convert_label(labels):
    # one-hot decoding
    if len(np.shape(labels)) == 1:
        return labels
    else:
        converted_labels = np.zeros(len(labels))
        for ix in range(len(labels)):
            converted_labels[ix] = np.argmax(labels[ix])
        return converted_labels


def train(train_df, milnet, criterion, optimizer, args, n_train, weight_kl):
    milnet.train()
    total_loss = 0
    for i, (bag_label, bag_feats) in enumerate(train_df):
        optimizer.zero_grad()
        if torch.isnan(bag_feats).sum() > 0:
            continue
        bag_label = bag_label.cuda()
        bag_feats = bag_feats.cuda()
        bag_feats = bag_feats.view(-1, args.feats_size)
        if args.model == 'dsmil':
            # refer to dsmil code
            ins_prediction, bag_prediction, _, _ = milnet(bag_feats, Train_flag=True, train_sample=n_train)
            max_prediction, _ = torch.max(ins_prediction, 0)
            bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
            loss = 0.5 * bag_loss + 0.5 * max_loss
            loss = loss + milnet.kl_loss() * weight_kl
        elif args.model == 'abmil':
            bag_prediction = milnet(bag_feats, Train_flag=True, train_sample=n_train)
            bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            loss = bag_loss
            loss = loss + milnet.kl_loss() * weight_kl
        elif args.model == 'transmil':
            output = milnet(h=bag_feats, Train_flag=True, train_sample=n_train)
            bag_prediction = output['logits']
            loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            loss = loss + milnet.kl_loss() * weight_kl
        else:
            raise NotImplementedError
        loss.backward()
        optimizer.step()
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
        for i, (bag_label, bag_feats) in enumerate(test_df):
            # bag_label, bag_feats = get_bag_feats_v2(test_feats[i], test_gts[i], args)
            bag_label = bag_label.cuda()
            bag_feats = bag_feats.cuda()
            bag_feats = bag_feats.view(-1, args.feats_size)
            if args.model == 'dsmil':
                ins_prediction, bag_prediction, _, _ = milnet(bag_feats, Train_flag=False, test_sample=n_test)
                max_prediction, _ = torch.max(ins_prediction, 0)
                bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
                max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
                loss = 0.5 * bag_loss + 0.5 * max_loss
            elif args.model == 'abmil':
                bag_prediction = milnet(bag_feats, Train_flag=False, test_sample=n_test)
                bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
                loss = bag_loss
            elif args.model == 'transmil':
                output = milnet(bag_feats)
                bag_prediction = output['logits']
                loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
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


def multi_label_roc(labels, predictions, num_classes):
    thresholds, thresholds_optimal, aucs = [], [], []
    if len(predictions.shape) == 1:
        predictions = predictions[:, None]
    if len(labels.shape) == 1:
        labels = labels[:, None]
    for c in range(0, num_classes):
        label = labels[:, c]
        prediction = predictions[:, c]
        fpr, tpr, threshold = roc_curve(label, prediction)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        c_auc = roc_auc_score(label, prediction)
        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)
    return aucs, thresholds, thresholds_optimal

def multi_label_roc_DTFD(labels, predictions, num_classes, pos_label=1):
    fprs = []
    tprs = []
    thresholds = []
    thresholds_optimal = []
    aucs = []
    if len(predictions.shape)==1:
        predictions = predictions[:, None]
    for c in range(0, num_classes):
        label = labels[:, c]
        if sum(label)==0:
            continue
        prediction = predictions[:, c]
        # print(label, prediction,label.shape, prediction.shape, labels.shape, predictions.shape)
        # dummy = []
        # for ii in range(len(prediction)):
        #     dummy.append(prediction[ii].tolist())
        # prediction = np.array(dummy)
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        c_auc = roc_auc_score(label, prediction)
        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)
    return aucs, thresholds, thresholds_optimal

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]


def main():
    parser = argparse.ArgumentParser(description='Train MIL Models with ReMix')
    parser.add_argument('--feats_size', default=512, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--lr', default=0.0002, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=50, type=int, help='Number of total training epochs')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID(s) [0]')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--dataset', default='BRACS_WSI', type=str,
                        choices=['Camelyon', 'Unitopatho', 'COAD', 'BRACS_WSI', 'NSCLC'], help='Dataset folder name')
    parser.add_argument('--task', default='binary', choices=['binary', 'staging'], type=str, help='Downstream Task')
    parser.add_argument('--model', default='dsmil', type=str,
                        choices=['dsmil', 'abmil', 'transmil', 'DTFD'], help='MIL model')
    # ReMix Parameters
    parser.add_argument('--num_prototypes', default=None, type=int, help='Number of prototypes per bag')
    parser.add_argument('--mode', default=None, type=str,
                        choices=['None', 'replace', 'append', 'interpolate', 'cov', 'joint'],
                        help='Augmentation method')
    parser.add_argument('--rate', default=0.5, type=float, help='Augmentation rate')

    # Utils
    parser.add_argument('--data_root', required=False, default='datasets', type=str, help='path to data root')
    parser.add_argument('--num_rep', default=1, type=int, help='Number of repeats')
    parser.add_argument('--num_workers', default=1, type=int, help='number rof workers')
    parser.add_argument('--wandb', action='store_true', help='Use wandb for logging')
    parser.add_argument('--distill', default='MaxMinS', type=str, help='Distillation method')
    parser.add_argument('--weight_path', default=None, type=str, help='Path to pretrained weights')
    parser.add_argument('--extractor', default='Resnet', type=str, help='Feature extractor')
    args = parser.parse_args()

    assert args.dataset in ['Camelyon', 'Unitopatho', 'COAD', 'BRACS_WSI', 'NSCLC'], 'Dataset not supported'
    # For Camelyon, we follow DSMIL to use binary labels: 1 for positive bags and 0 for negative bags.
    # For Unitopatho, we use one-hot encoding.
    if args.task == 'binary':
        args.num_classes = 1
    elif args.task == 'staging':
        args.num_classes = {'COAD': 4, 'BRACS_WSI': 3, 'BRCA': 4}[args.dataset]
    n_sample_test = [1]
    n_sample_train = [1]
    weight_kl = [1e-6]
    for t in range(args.num_rep):
        for n_train in n_sample_train:
            for n_test in n_sample_test:
                for kl in weight_kl:
                    config = {"lr": args.lr, "rep": t, "n_sample_train": n_train, "n_sample_test": n_test,
                              "kl_weight": kl}
                    # ckpt_pth = setup_logger(args, first_time)
                    # ckpt_pth = f'/home/yhchen/Documents/HDPMIL/datasets/{args.dataset}/ckpt.pth'
                    logging.info(f'current args: {args}')
                    logging.info(f'augmentation mode: {args.mode}')

                    # milnet = DP_Cluster(concentration=0.1,trunc=2,eta=1,batch_size=1,epoch=20, dim=512).cuda()
                    prior = {'horseshoe_scale': None, 'global_cauchy_scale': 1., 'weight_cauchy_scale': 1.,
                             'beta_rho_scale': -5.,
                             'log_tau_mean': None, 'log_tau_rho_scale': -5., 'bias_rho_scale': -5., 'log_v_mean': None,
                             'log_v_rho_scale': -5.}
                    # prepare model
                    if args.model == 'abmil':
                        # milnet = abmil.BClassifier(args.feats_size, args.num_classes).cuda()
                        # milnet = BClassifier(args.feats_size, args.num_classes).cuda()
                        milnet = ABMIL(args.feats_size, args.num_classes,  layer_type='HS', priors=prior,
                                       activation_type='relu').cuda()
                    elif args.model == 'dsmil':
                        # i_classifier = dsmil.FCLayer(in_size=args.feats_size, out_size=args.num_classes).cuda()
                        # b_classifier = dsmil.BClassifier(input_size=args.feats_size, output_class=args.num_classes, dropout_v=0).cuda()
                        # milnet = dsmil.MILNet(i_classifier, b_classifier).cuda()
                        milnet = DSMIL(args.feats_size, args.num_classes, layer_type='HS', priors=prior,
                                       activation_type='relu').cuda()
                    elif args.model == 'transmil':
                        milnet = TransMIL(args.feats_size, args.num_classes,  layer_type='HS', priors=prior,
                                          activation_type='relu').cuda()
                    elif args.model == 'DTFD':
                        mDim = args.feats_size // 2
                        DTFDclassifier = Classifier_1fc(mDim, args.num_classes, 0.0).cuda()
                        DTFDattention = Attention(mDim).cuda()
                        DTFDdimReduction = DimReduction(args.feats_size, mDim, numLayer_Res=0).cuda()
                        DTFDattCls = Attention_with_Classifier( L=mDim, n_classes=args.num_classes, \
                                                               droprate=0.0,layer_type='HS', priors=prior,
                                          activation_type='relu').cuda()
                        # DTFDattCls = Attention_with_Classifier(L=mDim, n_classes=args.num_classes, \
                        #                                        droprate=0.0).cuda()
                        milnet = [DTFDclassifier, DTFDattention, DTFDdimReduction, DTFDattCls]
                    else:
                        raise NotImplementedError

                    if args.num_classes == 1:
                        criterion = nn.BCEWithLogitsLoss()
                    else:
                        criterion = nn.CrossEntropyLoss()
                    if args.model == 'DTFD':
                        trainable_parameters = []
                        trainable_parameters += list(DTFDclassifier.parameters())
                        trainable_parameters += list(DTFDattention.parameters())
                        trainable_parameters += list(DTFDdimReduction.parameters())
                        optimizer_adam0 = torch.optim.Adam(trainable_parameters, lr=1e-4,
                                                           weight_decay=args.weight_decay)
                        scheduler0 = torch.optim.lr_scheduler.MultiStepLR(optimizer_adam0, [int(args.num_epochs / 2)],
                                                                          gamma=0.2)
                        optimizer_adam1 = torch.optim.Adam(DTFDattCls.parameters(), lr=1e-4, weight_decay=args.weight_decay)
                        scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer_adam1, [int(args.num_epochs / 2)],
                                                                          gamma=0.2)
                    else:
                        if args.model == 'transmil':
                            print('lood ahead optimizer in transmil....')
                            original_params = []
                            confounder_parms = []
                            for pname, p in milnet.named_parameters():
                                if ('confounder' in pname):
                                    confounder_parms += [p]
                                    print('confounders:', pname)
                                else:
                                    original_params += [p]
                            base_optimizer = RAdam([
                                {'params': original_params},
                                {'params': confounder_parms, ' weight_decay': 0.0001},
                            ],
                                lr=0.0002,
                                weight_decay=0.00001)
                            optimizer = Lookahead(base_optimizer)
                        else:
                            optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, betas=(0.5, 0.9),
                                                         weight_decay=args.weight_decay)
                        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)

                    train_path = os.path.join('datasets_csv', args.dataset,
                                              f'{args.task}_{args.dataset}_train' + '.csv')
                    train_path = pd.read_csv(train_path)
                    test_path = os.path.join('datasets_csv', args.dataset,
                                             f'{args.task}_{args.dataset}_testval' + '.csv')
                    test_path = pd.read_csv(test_path)

                    trainset = BagDataset(train_path, args)
                    train_loader = DataLoader(trainset, 1, shuffle=True, num_workers=args.num_workers, )
                    testset = BagDataset(test_path, args)
                    test_loader = DataLoader(testset, 1, shuffle=True, num_workers=args.num_workers)

                    config["rep"] = t
                    if args.wandb:
                        wandb.init(name=f'{args.task}_{args.dataset}_{args.model}_{args.extractor}',
                                   project='UAMIL',
                                   entity='yihangc',
                                   notes='',
                                   mode='online',  # disabled/online/offline
                                   config=config,
                                   tags=[])
                    best_acc = 0
                    for epoch in range(1, args.num_epochs + 1):
                        if args.model == 'DTFD':
                            start_time = time.time()
                            train_loss_bag = trainDTFD(args, train_loader, DTFDclassifier, \
                                                       DTFDdimReduction, DTFDattention, DTFDattCls, optimizer_adam0,
                                                       optimizer_adam1, epoch, criterion,kl_weight=kl)
                            print('epoch time:{}'.format(time.time() - start_time))
                            # test_loss_bag, avg_score, aucs, thresholds_optimal = test(test_loader, milnet, criterion, optimizer, args, log_path, epoch)
                            test_loss_bag, avg_score, aucs, thresholds_optimal = \
                                testDTFD(args, test_loader, DTFDclassifier, DTFDdimReduction, DTFDattention, \
                                         DTFDattCls, criterion,  epoch)

                            print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, AUC: ' %
                                  (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score) + '|'.join(
                                'class-{}>>{}'.format(*k) for k in enumerate(aucs)))
                            if scheduler0:
                                scheduler0.step()
                            scheduler1.step()
                            # current_score = (sum(aucs) + avg_score) / 2
                            # if current_score >= best_score:
                            #     best_score = current_score
                            #     save_name = os.path.join(save_path, str(run + 1) + '.pth')
                            #     tsave_dict = {
                            #         'classifier': DTFDclassifier.state_dict(),
                            #         'dim_reduction': DTFDdimReduction.state_dict(),
                            #         'attention': DTFDattention.state_dict(),
                            #         'att_classifier': DTFDattCls.state_dict()
                            #     }
                            #     torch.save(tsave_dict, save_name)
                            #     # torch.save(milnet.state_dict(), save_name)
                            #     # if args.dataset=='TCGA-lung':
                            #     #     print('Best model saved at: ' + save_name + ' Best thresholds: LUAD %.4f, LUSC %.4f' % (thresholds_optimal[0], thresholds_optimal[1]))
                            #     # else:
                            #     with open(log_path, 'a+') as log_txt:
                            #         info = 'Best model saved at: ' + save_name + '\n'
                            #         log_txt.write(info)
                            #         info = 'Best thresholds ===>>> ' + '|'.join(
                            #             'class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal)) + '\n'
                            #         log_txt.write(info)
                            #     print('Best model saved at: ' + save_name)
                            #     print('Best thresholds ===>>> ' + '|'.join(
                            #         'class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal)))
                        else:
                            train_loss_bag = train(train_loader, milnet, criterion, optimizer, args, n_train, weight_kl=kl)
                            if args.num_classes == 1:
                                precision, recall, accuracy, f1, avg, auc = test(test_loader, milnet, criterion, args,
                                                                                 n_test)
                                print(f'pre:{precision},recall:{recall},acc:{accuracy},f1:{f1},auc:{auc}.')
                                if args.wandb:
                                    wandb.log({'train_loss': train_loss_bag, 'precision': precision, 'recall': recall,
                                               'accuracy': accuracy, 'f1': f1,
                                               'avg': avg, 'auc': auc})
                            else:
                                accuracy, f1, auc = test(test_loader, milnet, criterion, args, n_test)
                                print(f'acc:{accuracy},f1:{f1},auc:{auc}.')
                                if args.wandb:
                                    wandb.log({'train_loss': train_loss_bag, 'accuracy': accuracy, 'f1': f1, 'auc': auc})
                            logging.info('Epoch [%d/%d] train loss: %.4f' % (epoch, args.num_epochs, train_loss_bag))
                            if args.model != 'transmil':
                                scheduler.step()
                            # if accuracy >= best_acc:
                            #     print('saving model...')
                            #     best_acc = accuracy
                            #     torch.save(milnet.state_dict(), ckpt_pth)
                    if args.wandb:
                        wandb.finish()

        # precision, recall, accuracy, f1, avg, auc = test(test_feats, test_labels, milnet, criterion, args)
        # torch.save(milnet.state_dict(), ckpt_pth)
        # logging.info('Final model saved at: ' + ckpt_pth)
        # logging.info(f'Precision, Recall, Accuracy, Avg, AUC')
        # logging.info(f'{precision*100:.2f} {recall*100:.2f} {accuracy*100:.2f} {avg*100:.2f} {auc*100:.2f}')
        first_time = False


def trainDTFD(args, train_df, classifier, dimReduction, attention, UClassifier, optimizer0, optimizer1, epoch, \
              criterion=None, numGroup=4, total_instance=4, kl_weight=1e-6):
    distill = args.distill
    # SlideNames_list, mFeat_list, Label_dict = mDATA_list
    total_loss = 0
    classifier.train()
    if not args.weight_path:
        dimReduction.train()
    else:
        dimReduction.eval()
    attention.train()
    UClassifier.train()

    instance_per_group = total_instance // numGroup

    numSlides = len(train_df)

    tIDX = list(range(numSlides))
    random.shuffle(tIDX)

    for i, (bag_label, bag_feats) in enumerate(train_df):
        # if i < 265: continue
        bag_label = bag_label.cuda()
        bag_feats = bag_feats.cuda()
        bag_feats = bag_feats.view(-1, args.feats_size)  # n x feat_dim
        tslideLabel = bag_label

        slide_pseudo_feat = []
        slide_sub_preds = []
        slide_sub_labels = []

        tfeat_tensor = bag_feats

        feat_index = list(range(tfeat_tensor.shape[0]))
        random.shuffle(feat_index)
        index_chunk_list = np.array_split(np.array(feat_index), numGroup)
        index_chunk_list = [sst.tolist() for sst in index_chunk_list]

        for tindex in index_chunk_list:
            slide_sub_labels.append(tslideLabel)
            subFeat_tensor = torch.index_select(tfeat_tensor, dim=0, index=torch.LongTensor(tindex).cuda())
            tmidFeat = dimReduction(subFeat_tensor)
            tAA = attention(tmidFeat).squeeze(0)
            tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
            tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs
            tPredict, bg_feat0, Att_s0 = classifier(tattFeat_tensor)  ### 1 x 2
            slide_sub_preds.append(tPredict)

            patch_pred_logits = get_cam_1d(classifier, tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n
            patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
            patch_pred_softmax = torch.sigmoid(patch_pred_logits)  ## n x cls

            _, sort_idx = torch.sort(patch_pred_softmax[:, -1], descending=True)
            topk_idx_max = sort_idx[:instance_per_group].long()
            topk_idx_min = sort_idx[-instance_per_group:].long()
            topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)

            MaxMin_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)  ##########################
            max_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx_max)
            # af_inst_feat = tattFeat_tensor
            af_inst_feat = bg_feat0

            if distill == 'MaxMinS':
                slide_pseudo_feat.append(MaxMin_inst_feat)
            elif distill == 'MaxS':
                slide_pseudo_feat.append(max_inst_feat)
            elif distill == 'AFS':
                slide_pseudo_feat.append(af_inst_feat)

        slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)  ### numGroup x fs

        ## optimization for the first tier

        slide_sub_preds = torch.cat(slide_sub_preds, dim=0)  ### numGroup x fs
        slide_sub_labels = torch.cat(slide_sub_labels, dim=0)  ### numGroup
        loss0 = criterion(slide_sub_preds, slide_sub_labels).mean()
        grad_clipping = 5.0
        if optimizer0:
            optimizer0.zero_grad()
            loss0.backward(retain_graph=True)
            if not args.weight_path:
                torch.nn.utils.clip_grad_norm_(dimReduction.parameters(), grad_clipping)
            torch.nn.utils.clip_grad_norm_(attention.parameters(), grad_clipping)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), grad_clipping)
            optimizer0.step()

        ## optimization for the second tier
        gSlidePred, bg_feat, Att_s1 = UClassifier(slide_pseudo_feat.detach())
        # gSlidePred = UClassifier(slide_pseudo_feat)
        loss1 = criterion(gSlidePred, tslideLabel).mean() + kl_weight * UClassifier.kl_loss()
        # loss1 = criterion(gSlidePred, tslideLabel).mean()
        optimizer1.zero_grad()
        loss1.backward()
        UClassifier.analytic_update()
        torch.nn.utils.clip_grad_norm_(UClassifier.parameters(), grad_clipping)
        optimizer1.step()
        total_loss = total_loss + loss0.item() + loss1.item()


        sys.stdout.write('\r Training bag [{:}/{:}] bag loss: {:.4f}'. \
                         format(i, len(train_df), loss0.item() + loss1.item()))

    return total_loss / len(train_df)


def testDTFD(args, test_df, classifier, dimReduction, attention, UClassifier, \
             criterion, epoch, numGroup=4, total_instance=4):
    distill = args.distill
    classifier.eval()
    attention.eval()
    dimReduction.eval()
    UClassifier.eval()

    instance_per_group = total_instance // numGroup
    gPred_0 = torch.FloatTensor().cuda()
    gt_0 = torch.LongTensor().cuda()
    gPred_1 = torch.FloatTensor().cuda()
    gt_1 = torch.LongTensor().cuda()
    total_loss = 0
    test_labels = []
    test_predictions = []
    Tensor = torch.cuda.FloatTensor
    with torch.no_grad():
        for i, (bag_label, bag_feats) in enumerate(test_df):
            label = bag_label.numpy()
            bag_label = bag_label.cuda()
            bag_feats = bag_feats.cuda()
            bag_feats = bag_feats.view(-1, args.feats_size)

            tslideLabel = bag_label
            tfeat = bag_feats
            midFeat = dimReduction(tfeat)
            AA = attention(midFeat, isNorm=False).squeeze(0)  ## N
            allSlide_pred_softmax = []
            num_MeanInference = 1
            for jj in range(num_MeanInference):

                feat_index = list(range(tfeat.shape[0]))
                random.shuffle(feat_index)
                index_chunk_list = np.array_split(np.array(feat_index), numGroup)
                index_chunk_list = [sst.tolist() for sst in index_chunk_list]

                slide_d_feat = []
                slide_sub_preds = []
                slide_sub_labels = []

                for tindex in index_chunk_list:
                    slide_sub_labels.append(tslideLabel)
                    idx_tensor = torch.LongTensor(tindex).cuda()
                    tmidFeat = midFeat.index_select(dim=0, index=idx_tensor)

                    tAA = AA.index_select(dim=0, index=idx_tensor)
                    tAA = torch.softmax(tAA, dim=0)  # n
                    tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
                    tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs

                    tPredict, bg_feat0, Att_s0 = classifier(tattFeat_tensor)  ### 1 x 2
                    slide_sub_preds.append(tPredict)

                    patch_pred_logits = get_cam_1d(classifier, tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n
                    patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
                    # patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  ## n x cls
                    patch_pred_softmax = torch.sigmoid(patch_pred_logits)  ## n x cls

                    _, sort_idx = torch.sort(patch_pred_softmax[:, -1], descending=True)

                    if distill == 'MaxMinS':
                        topk_idx_max = sort_idx[:instance_per_group].long()
                        topk_idx_min = sort_idx[-instance_per_group:].long()
                        topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)
                        d_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
                        slide_d_feat.append(d_inst_feat)
                    elif distill == 'MaxS':
                        topk_idx_max = sort_idx[:instance_per_group].long()
                        topk_idx = topk_idx_max
                        d_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
                        slide_d_feat.append(d_inst_feat)
                    elif distill == 'AFS':
                        # slide_d_feat.append(tattFeat_tensor)
                        slide_d_feat.append(bg_feat0)

                slide_d_feat = torch.cat(slide_d_feat, dim=0)
                slide_sub_preds = torch.cat(slide_sub_preds, dim=0)
                slide_sub_labels = torch.cat(slide_sub_labels, dim=0)

                gPred_0 = torch.cat([gPred_0, slide_sub_preds], dim=0)
                gt_0 = torch.cat([gt_0, slide_sub_labels], dim=0)
                loss0 = criterion(slide_sub_preds, slide_sub_labels).mean()
                # test_loss0.update(loss0.item(), numGroup)

                gSlidePred, bag_feat, Att_s1 = UClassifier(slide_d_feat)
                # allSlide_pred_softmax.append(torch.softmax(gSlidePred, dim=1))
                allSlide_pred_softmax.append(torch.sigmoid(gSlidePred))  # [1,1]

            allSlide_pred_softmax = torch.cat(allSlide_pred_softmax, dim=0)
            allSlide_pred_softmax = torch.mean(allSlide_pred_softmax, dim=0).unsqueeze(0)
            gPred_1 = torch.cat([gPred_1, allSlide_pred_softmax], dim=0)
            gt_1 = torch.cat([gt_1, tslideLabel], dim=0)

            # loss1 = F.nll_loss(allSlide_pred_softmax, tslideLabel)
            loss1 = criterion(allSlide_pred_softmax, tslideLabel)
            # test_loss1.update(loss1.item(), 1)

            total_loss = total_loss + loss0.item() + loss1.item()

            sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (i, len(test_df), loss0.item()))
            sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (i, len(test_df), loss1.item()))
            test_labels.extend(label)
            test_predictions.extend([allSlide_pred_softmax.squeeze().cpu().numpy()])

    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)

    auc_value, _, thresholds_optimal = multi_label_roc_DTFD(test_labels, test_predictions, args.num_classes, pos_label=1)
    test_predictions_ = test_predictions > 0.5
    acc = accuracy_score(test_labels, test_predictions_)
    cls_report = classification_report(test_labels, test_predictions_, digits=4)

    print('Accuracy', acc)
    print('\n', cls_report)

    # chosing threshold
    if args.num_classes == 1:
        res = binary_metrics_fn(test_labels, test_predictions,
                                metrics=['accuracy', 'precision', 'recall', 'roc_auc', 'f1'])
        if args.wandb:
            wandb.log(
                {'precision': res['precision'], 'recall': res['recall'], 'accuracy': res['accuracy'], 'f1': res['f1'],
                 'auc': res['roc_auc']})
        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions >= thresholds_optimal[0]] = 1
        class_prediction_bag[test_predictions < thresholds_optimal[0]] = 0
        test_predictions = class_prediction_bag
        test_labels = np.squeeze(test_labels)
        print(confusion_matrix(test_labels, test_predictions))

    else:
        for i in range(args.num_classes):
            class_prediction_bag = copy.deepcopy(test_predictions[:, i])
            class_prediction_bag[test_predictions[:, i] >= thresholds_optimal[i]] = 1
            class_prediction_bag[test_predictions[:, i] < thresholds_optimal[i]] = 0
            test_predictions[:, i] = class_prediction_bag
            print(confusion_matrix(test_labels[:, i], test_predictions[:, i]))


    bag_score = 0
    # average acc of all labels
    for i in range(0, len(test_df)):
        bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score
    avg_score = bag_score / len(test_df)  # ACC
    cls_report = classification_report(test_labels, test_predictions, digits=4)

    # print(confusion_matrix(test_labels,test_predictions))
    print('\n dsmil-metrics: multi-label Accuracy:{:.2f}, AUC:{:.2f}'.format(avg_score * 100,
                                                                             sum(auc_value) / len(auc_value) * 100))
    print('\n', cls_report)

    return total_loss / len(test_df), avg_score, auc_value, thresholds_optimal

def test_generalizability(args):
    if args.model == 'abmil':
        milnet = abmil.BClassifier(args.feats_size, args.num_classes).cuda()
        weights = torch.load('REMIX_COAD2BRCA.pth')
    elif args.model == 'dsmil':
        i_classifier = dsmil.FCLayer(in_size=args.feats_size, out_size=args.num_classes).cuda()
        b_classifier = dsmil.BClassifier(input_size=args.feats_size, output_class=args.num_classes, dropout_v=0).cuda(
            )
        milnet = dsmil.MILNet(i_classifier, b_classifier).cuda()
        weights = torch.load('DSMIL_COAD2BRCA.pth')
    milnet.load_state_dict(weights, strict=True)
    test_labels_pth = f'datasets/BRCA/binary_BRCA_testval_label.npy'
    test_feats = open(f'datasets/BRCA/binary_BRCA_testval.txt', 'r').readlines()
    test_feats = np.array(test_feats)
    criterion = nn.BCEWithLogitsLoss()
    test_labels = np.load(test_labels_pth)
    test_labels = torch.Tensor(test_labels).cuda()

    precision, recall, accuracy, f1, avg, auc = test(test_feats, test_labels, milnet, criterion, args)
    print(f'pre:{precision},recall:{recall},acc:{accuracy},f1:{f1},auc:{auc}.')


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()