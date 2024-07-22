import torch
import torch.nn as nn
from BNN.models import ABMIL,BClassifier
from dataset import BagDataset
import argparse
import pandas as pd
from torch.utils.data import DataLoader
def main():
    parser = argparse.ArgumentParser(description='Train MIL Models with ReMix')
    parser.add_argument('--feats_size', default=512, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--dataset', default='Camelyon', type=str,
                        choices=['Camelyon', 'Unitopatho', 'COAD', 'BRACS_WSI', 'NSCLC'], help='Dataset folder name')
    parser.add_argument('--task', default='binary', choices=['binary', 'staging'], type=str, help='Downstream Task')
    parser.add_argument('--model', default='abmil', type=str,
                        choices=[ 'abmil', 'abuamil'], help='MIL model')
    # Utils
    parser.add_argument('--data_root', required=False, default='datasets', type=str, help='path to data root')
    parser.add_argument('--weight_path', required=True, default=None, type=str, help='Path to pretrained weights')
    parser.add_argument('--extractor', default='dsmil', type=str, help='Feature extractor')
    parser.add_argument('--num_classes', default=1, type=int, help='Number of classes')
    args = parser.parse_args()
    if args.model == 'abmil':
        milnet = BClassifier(args.feats_size,args.num_classes).cuda()
    elif args.model == 'abuamil':
        milnet = BClassifier(args.feats_size).cuda()
    state_dict_weights = torch.load(f'{args.weight_path}')
    milnet.load_state_dict(state_dict_weights)
    milnet.eval()
    sample_path = 'annotation_tif/samples.csv'
    sample_path = pd.read_csv(sample_path)
    dataset = BagDataset(sample_path, args)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        for i, (_, bag_feats, name) in enumerate(dataloader):
            A = milnet.get_pred(bag_feats)
            # A = milnet.get_attention(bag_feats)
            pred = (A.view(-1)).cpu().numpy()
            slide_name = name[0].split('/')[-1][:8]
            if 'test' not in slide_name:
                coor_pth = f'Feats/Camelyon/simclr_files_256_v2/training/{slide_name}/c_idx.txt'
            else:
                coor_pth = f'Feats/Camelyon/simclr_files_256_v2/testing/{slide_name}/c_idx.txt'
            with open(coor_pth) as f:
                coor = f.readlines()
            X = []
            Y = []
            for item in coor:
                X.append(int(item.split('\t')[0]) * 256)
                Y.append(int(item.split('\t')[1]) * 256)
            coor_prob_info = {'X': X, 'Y': Y, 'logit': pred}
            coor_prob_info = pd.DataFrame(coor_prob_info)
            coor_prob_info.to_csv(
                f'Attention/{args.model}_{slide_name}_coor_logit.csv')

if __name__ == '__main__':
    main()