'''
Written by SeuTao
'''
import os
import time
import numpy as np
import torch
from setting import parse_opts
from torch.utils.data import DataLoader
from datasets.TReNDs import TReNDsDataset
from model import generate_model
from tqdm import tqdm
import torch.nn.functional as F


def get_features(model, x):
    x = model.module.conv1(x)
    x = model.module.bn1(x)
    x = model.module.relu(x)
    x = model.module.maxpool(x)
    x = model.module.layer1(x)
    x = model.module.layer2(x)
    x = model.module.layer3(x)
    x = model.module.layer4(x)

    x = F.adaptive_avg_pool3d(x, (1, 1, 1))
    emb_3d = x.view((-1, model.module.fea_dim))
    return emb_3d


def test_features(data_loader, model, sets, save_path):
    # settings
    print("validation")
    model.eval()

    y_features = []
    ids_all = []
    with torch.no_grad():
        for i, batch_data in enumerate(tqdm(data_loader)):
                # getting data batch
                ids, volumes = batch_data
                if not sets.no_cuda:
                    volumes = volumes.cuda()

                features = get_features(model, volumes)
                print(features.shape)
                y_features.append(features.data.cpu().numpy())
                ids_all += ids
                if i > 10:
                    break

    y_features = np.concatenate(y_features, axis=0)
    np.savez_compressed(save_path,
                        y_features = y_features,
                        ids = ids_all)
    print(y_features.shape)

if __name__ == '__main__':

    sets = parse_opts()
    sets.no_cuda = False
    #sets.resume_path = None
    #sets.pretrain_path = None

    #sets.batch_size = 32
    sets.num_workers = 16
    #sets.model_depth = 10
    #sets.resnet_shortcut = 'A'

    #sets.n_epochs = 50
    sets.fold_index = 0

    sets.model_name = r'prue_3dconv'
    sets.save_folder = r'./TReNDs/{}/' \
                       r'models_{}_{}_{}_fold_{}'.format(sets.model_name, 'resnet',sets.model_depth,sets.resnet_shortcut,sets.fold_index)

    if not os.path.exists(sets.save_folder):
        os.makedirs(sets.save_folder)

    # getting model
    torch.manual_seed(sets.manual_seed)
    model, parameters = generate_model(sets)
    print(model)

    # optimizer
    def get_optimizer(net):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=3e-4, betas=(0.9, 0.999), eps=1e-08)
        def ajust_lr(optimizer, epoch):
                if epoch < 24 :
                    lr = 3e-4
                elif epoch < 36:
                    lr = 1e-4
                else:
                    lr = 1e-5

                for p in optimizer.param_groups:
                    p['lr'] = lr
                return lr

        rate = ajust_lr(optimizer, 0)
        return  optimizer, ajust_lr

    optimizer, ajust_lr = get_optimizer(model)
    # train from resume
    if sets.resume_path:
        if os.path.isfile(sets.resume_path):
            print("=> loading checkpoint '{}'".format(sets.resume_path))
            checkpoint = torch.load(sets.resume_path)
            model.load_state_dict(checkpoint['state_dict'])

    # getting data
    sets.phase = 'train'
    if sets.no_cuda:
        sets.pin_memory = False
    else:
        sets.pin_memory = True


    valid_dataset = TReNDsDataset(mode='valid_test')
    valid_loader = DataLoader(valid_dataset, batch_size=sets.batch_size,
                             shuffle=False, num_workers=sets.num_workers,
                             pin_memory=sets.pin_memory, drop_last=False)
    test_features(valid_loader, model, sets, os.path.join(sets.save_folder, 'features.npz'))
