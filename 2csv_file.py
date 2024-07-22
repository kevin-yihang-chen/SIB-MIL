import numpy as np
import pandas as pd
import os
import os.path
curr_path = os.path.dirname(__file__)
data_root = 'datasets_csv'
dataset = 'BRCA'
task = 'binary'
if task == 'binary':
    train_labels_pth = f'{data_root}/{dataset}/{task}_{dataset}_train_label.npy'
    test_labels_pth = f'{data_root}/{dataset}/{task}_{dataset}_testval_label.npy'
    test_feats = open(f'{data_root}/{dataset}/{task}_{dataset}_testval.txt', 'r').readlines()
    train_feats = open(f'{data_root}/{dataset}/{task}_{dataset}_train.txt', 'r').readlines()
    train_labels, test_labels = np.load(train_labels_pth), np.load(test_labels_pth)
    train_dict = {'label': train_labels, 'slide': train_feats}
    test_dict = {'label': test_labels, 'slide': test_feats}
    train_df = pd.DataFrame(train_dict)
    test_df = pd.DataFrame(test_dict)
    os.makedirs(f'datasets_csv/{dataset}', exist_ok=True)
    train_df.to_csv(f'datasets_csv/{dataset}/{task}_{dataset}_train.csv', index=False)
    test_df.to_csv(f'datasets_csv/{dataset}/{task}_{dataset}_testval.csv', index=False)
elif task == 'OOD':
    in_dataset = 'Camelyon'
    out_dataset = 'PRAD'
    in_test_feats = open(f'{data_root}/{in_dataset}/binary_{in_dataset}_testval.txt', 'r').readlines()
    out_test_feats = open(f'{data_root}/{out_dataset}/binary_{out_dataset}_testval.txt', 'r').readlines()
    in_labels = np.zeros(len(in_test_feats))
    out_labels = np.ones(len(out_test_feats))
    in_test_feats.extend(out_test_feats)
    labels = np.concatenate([in_labels, out_labels])
    OOD_dict = {'label': labels, 'slide': in_test_feats}
    OOD_df = pd.DataFrame(OOD_dict)
    os.makedirs(f'datasets_csv/{in_dataset}', exist_ok=True)
    OOD_df.to_csv(f'datasets_csv/{in_dataset}/{task}_{in_dataset}_{out_dataset}.csv', index=False)