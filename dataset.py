from torch.utils.data import Dataset
import torch
import numpy as np
import os
import glob
import pandas as pd
import random
class InstanceDataset(Dataset):
    def __init__(self, ood_dataset, args) -> None:
        super(InstanceDataset).__init__()
        self.ood_dataset = ood_dataset
        self.args = args
        self.in_dataframe = pd.read_csv('datasets_csv/Camelyon/binary_Camelyon_testval.csv')
        self.out_datalist = glob.glob(f'Feats/{ood_dataset}/*/*/features.pt')
    def get_instance_feats(self, csv_file_df):
        feats = csv_file_df.iloc[1].split('\n')[0]
        if self.args.extractor == 'Kimia':
            feats = feats.replace('/data1/WSI/Patches/Features/COAD', 'Feats/COAD')
            feats = feats.replace('/data1/WSI/Patches/Features/PRAD', 'Feats/PRAD')
            feats = feats.replace('/data1/WSI/Patches/Features/Camelyon16', 'Feats/Camelyon')
            feats = feats.replace('simclr_files_256_v0', 'Camelyon16_Tissue_Kimia_20x')
            feats_csv_path = feats + '/features.pt'
        feats = torch.load(feats_csv_path).cuda()
        feats = feats[np.random.permutation(len(feats))]
        feats, labels = self.construct_ood_feats(feats)
        permutation = np.random.permutation(len(feats))
        feats = feats[permutation]
        labels = labels[permutation]
        return labels, feats

    def construct_ood_feats(self, feats):
        ood_instances = random.choices(self.out_datalist, k=10)
        ood_feats = torch.load(ood_instances[0])
        for i in range(1, len(ood_instances)):
            ood_feats = torch.cat((ood_feats, torch.load(ood_instances[i])), dim=0)
        num_ood_instances = round(feats.shape[0]*self.args.ood_ratio)
        ood_feats = ood_feats[np.random.permutation(len(ood_feats))][:num_ood_instances]
        feats = torch.cat((feats[num_ood_instances:],ood_feats),dim=0)
        labels = torch.cat((torch.zeros(feats.shape[0]-num_ood_instances),torch.ones(num_ood_instances)),dim=0)
        return feats, labels

    def __len__(self):
        return len(self.in_dataframe)
    def __getitem__(self, idx):
        labels, feats = self.get_instance_feats(self.in_dataframe.iloc[idx])
        return labels, feats
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
                feats = feats.replace('/data1/WSI/Patches/Features/COAD', 'Feats/COAD')
                feats = feats.replace('/data1/WSI/Patches/Features/BRCA', 'Feats/BRCA')
                feats = feats.replace('/data1/WSI/Patches/Features/PRAD', 'Feats/PRAD')
                feats = feats.replace('/data1/WSI/Patches/Features/Camelyon16', 'Feats/Camelyon')
                feats = feats.replace('simclr_files_256_v0', 'Camelyon16_Tissue_Kimia_20x')
                feats_csv_path = feats + '/features.pt'
            else:
                feats = csv_file_df.iloc[1].split('\n')[0]
                feats = feats.replace('/data1/WSI/Patches/Features/COAD', 'Feats/COAD')
                feats = feats.replace('/data1/WSI/Patches/Features/BRCA', 'Feats/BRCA')
                feats_csv_path = feats + '/features.pt'
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
                # feats = feats.replace('simclr_files_256_v0', 'Camelyon16_Tissue_ResNet_20x')
                # feats_csv_path = feats + '/features.pt'
                feats = feats.replace('simclr_files_256_v0', 'Camelyon16_ImageNet')
                feats = feats.replace('training', 'train')
                feats = feats.replace('testing', 'test')
                feats_csv_path = feats + '.pt'
            else:
                feats_csv_path = csv_file_df.iloc[1]
                feats_csv_path = feats_csv_path.split('\n')[0] + '/features.pt'
        elif args.extractor == 'dsmil':
            if args.dataset == 'Camelyon':
                feats = csv_file_df.iloc[1].split('\n')[0]
                feats = feats.replace('/data1/WSI/Patches/Features/Camelyon16', 'Feats/Camelyon')
                feats = feats.replace('simclr_files_256_v0', 'simclr_files_256_v2')
                feats_csv_path = feats + '/features.pt'
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        feats = torch.load(feats_csv_path).cuda()
        feats = feats[np.random.permutation(len(feats))]
        # label = np.zeros(args.num_classes)
        # if args.num_classes == 1:
        #     label[0] = csv_file_df.iloc[0]
        # else:
        #     # if int(csv_file_df.iloc[1]) <= (len(label) - 1):
        #     #     label[int(csv_file_df.iloc[1])] = 1
        #     label = csv_file_df.iloc[0]
        label = csv_file_df.iloc[0]
        label = torch.tensor(np.array(label),dtype=float).unsqueeze(dim=0)

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
        return label, feats, self.train_path.iloc[idx].iloc[1]

    def __len__(self):
        return len(self.train_path)