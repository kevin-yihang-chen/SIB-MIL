from torch.utils.data import Dataset
import torch
import numpy as np
import os
class BagDataset(Dataset):
    def __init__(self, train_path, args) -> None:
        super(BagDataset).__init__()
        self.train_path = train_path
        self.args = args
        # self.database = redis.Redis(host='localhost', port=6379)

    def get_bag_feats(self, csv_file_df, args):
        # if args.dataset == 'TCGA-lung-default':
        #     feats_csv_path = 'datasets/tcga-dataset/tcga_lung_data_feats/' + csv_file_df.iloc[0].split('/')[1] + '.csv'
        if args.extractor == 'Kimia':
            if args.dataset == 'NSCLC':
                if 'LUAD' in csv_file_df.iloc[1]:
                    pre_path = 'Feats'
                    slide_name = csv_file_df.iloc[1].split('/')[-1].split('\n')[0]
                    slide_name = slide_name.split('.')[0]
                    feats_csv_path = pre_path + '/LUAD/LUAD_Diagnostic_Kimia_20x/' + slide_name + '/features.pt'
                else:
                    pre_path = 'Feats'
                    slide_name = csv_file_df.iloc[1].split('/')[-1].split('\n')[0]
                    slide_name = slide_name.split('.')[0]
                    feats_csv_path = pre_path + '/LUSC/LUSC_Diagnostic_Kimia_20x/' + slide_name + '/features.pt'
            elif args.dataset == 'Camelyon':
                feats = csv_file_df.iloc[1].split('\n')[0]
                feats = feats.replace('/data1/WSI/Patches/Features/Camelyon16', 'Feats/Camelyon')
                # feats = feats.replace('v0', 'v2')
                # feats_csv_path = feats + '/features.pt'
                feats = feats.replace('simclr_files_256_v0', 'Camelyon16_Tissue_Kimia_20x')
                feats_csv_path = feats + '/features.pt'
            else:
                feats_csv_path = csv_file_df.iloc[1]
                feats_csv_path = feats_csv_path.split('\n')[0] + '/features.pt'
        elif args.extractor == 'Resnet':
            if args.dataset == 'NSCLC':
                if 'LUAD' in csv_file_df.iloc[1]:
                    pre_path = 'Feats'
                    slide_name = csv_file_df.iloc[1].split('/')[-1].split('\n')[0]
                    slide_name = slide_name.split('.')[0]
                    feats_csv_path = pre_path + '/LUAD/LUAD_Diagnostic_ResNet_20x/' + slide_name + '/features.pt'
                else:
                    pre_path = 'Feats'
                    slide_name = csv_file_df.iloc[1].split('/')[-1].split('\n')[0]
                    slide_name = slide_name.split('.')[0]
                    feats_csv_path = pre_path + '/LUSC/LUSC_Diagnostic_ResNet_20x/' + slide_name + '/features.pt'
            elif args.dataset == 'Camelyon':
                feats = csv_file_df.iloc[1].split('\n')[0]
                feats = feats.replace('/data1/WSI/Patches/Features/Camelyon16', 'Feats/Camelyon')
                # feats = feats.replace('v0', 'v2')
                # feats_csv_path = feats + '/features.pt'
                feats = feats.replace('simclr_files_256_v0', 'Camelyon16_Tissue_ResNet_20x')
                feats_csv_path = feats + '/features.pt'
            else:
                feats_csv_path = csv_file_df.iloc[1]
                feats_csv_path = feats_csv_path.split('\n')[0] + '/features.pt'

        feats = torch.load(feats_csv_path).cuda()
        feats = feats[np.random.permutation(len(feats))]
        label = np.zeros(args.num_classes)
        if args.num_classes == 1:
            label[0] = csv_file_df.iloc[0]
        else:
            if int(csv_file_df.iloc[1]) <= (len(label) - 1):
                label[int(csv_file_df.iloc[1])] = 1
        label = torch.tensor(np.array(label))

        return label, feats

    def dropout_patches(self, feats, p):
        idx = np.random.choice(np.arange(feats.shape[0]), int(feats.shape[0] * (1 - p)), replace=False)
        sampled_feats = np.take(feats, idx, axis=0)
        pad_idx = np.random.choice(np.arange(sampled_feats.shape[0]), int(feats.shape[0] * p), replace=False)
        pad_feats = np.take(sampled_feats, pad_idx, axis=0)
        sampled_feats = np.concatenate((sampled_feats, pad_feats), axis=0)
        return sampled_feats

    def __getitem__(self, idx):
        label, feats = self.get_bag_feats(self.train_path.iloc[idx], self.args)
        return label, feats

    def __len__(self):
        return len(self.train_path)