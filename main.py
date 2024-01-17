# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:59:14 2020

@author: HQ Xie
"""
import os
import argparse
import time
import json
import torch
import random
import torch.nn as nn
import numpy as np
from utils import SNR_to_noise, initNetParams, train_step, val_step, train_mi
from utils import train_step_mlp, train_mi_mlp, val_step_mlp
from dataset import EurDataset, collate_data, TimeseriesDataset
from models.transceiver import DeepSC,DeepSC_MLP
from models.mutual_info import Mine
from torch.utils.data import DataLoader
from tqdm import tqdm

parser = argparse.ArgumentParser()
#parser.add_argument('--data-dir', default='data/train_data.pkl', type=str)
# parser.add_argument('--vocab-file', default='europarl/vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='checkpoints/deepsc-Rayleigh', type=str)
parser.add_argument('--channel', default='Rayleigh', type=str, help = 'Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--MAX-LENGTH', default=96, type=int)
parser.add_argument('--MIN-LENGTH', default=96, type=int)
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--epochs', default=80, type=int)
parser.add_argument('--model', default="mlp", type=str)
parser.add_argument('--d_input', default=96, type=int)
parser.add_argument('--d_output', default=96, type=int)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def validate(epoch, args, net):
    test_eur = TimeseriesDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True)
    net.eval()
    pbar = tqdm(test_iterator)
    total = 0
    with torch.no_grad():
        for sents in pbar:
            sents = sents.float().to(device)
            if args.model == "transformer":
                loss = val_step(net, sents, sents, 0.1, pad_idx,
                                criterion, args.channel)
            else:
                loss = val_step_mlp(net, sents, sents, 0.1,
                                criterion, args.channel)

            total += loss
            pbar.set_description(
                'Epoch: {}; Type: VAL; Loss: {:.5f}'.format(
                    epoch + 1, loss
                )
            )

    return total/len(test_iterator)


def train(epoch, args, net, mi_net=None):
    train_eur= TimeseriesDataset('train')
    train_iterator = DataLoader(train_eur, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True)
    pbar = tqdm(train_iterator)

    noise_std = np.random.uniform(SNR_to_noise(5), SNR_to_noise(10), size=(1))

    for sents in pbar:
        sents = sents.float().to(device)
        
        if mi_net is not None:
            if args.model == 'transformer':
                mi = train_mi(net, mi_net, sents, 0.1, pad_idx, mi_opt, args.channel)
            
                loss = train_step(net, sents, sents, 0.1, pad_idx,
                              optimizer, criterion, args.channel, mi_net)
            elif args.model == 'mlp':
                mi = train_mi_mlp(net, mi_net, sents, 0.1, mi_opt, args.channel)
                loss = train_step_mlp(net, sents, sents, 0.1,
                              optimizer, criterion, args.channel, mi_net)
            pbar.set_description(
                'Epoch: {};  Type: Train; Loss: {:.5f}; MI {:.5f}'.format(
                    epoch + 1, loss, mi
                )
            )
            exit()
        else:
            if args.model == 'transformer':
                loss = train_step(net, sents, sents, noise_std[0], pad_idx,
                                optimizer, criterion, args.channel)
            else:
                loss = train_step_mlp(net, sents, sents, 0.1,
                              optimizer, criterion, args.channel)
            pbar.set_description(
                'Epoch: {};  Type: Train; Loss: {:.5f}'.format(
                    epoch + 1, loss
                )
            )


if __name__ == '__main__':
    # setup_seed(10)
    args = parser.parse_args()

    pad_idx = -999
    """ define optimizer and loss function """
    if args.model == "transformer":
        deepsc = DeepSC(args.num_layers, args.d_input,
                        args.d_output,args.d_model, 
                        args.num_heads,args.dff, 
                        0.1).to(device)
    elif args.model == "mlp":
        deepsc = DeepSC_MLP(args.d_input,
                            args.d_output).to(device)
    else:
        raise ValueError('No model.')
    mi_net = Mine().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(deepsc.parameters(),
                                 lr=1e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay = 5e-4)
    mi_opt = torch.optim.Adam(mi_net.parameters(), lr=1e-3)
    #opt = NoamOpt(args.d_model, 1, 4000, optimizer)
    initNetParams(deepsc)
    for epoch in range(args.epochs):
        start = time.time()
        record_acc = 10

        train(epoch, args, deepsc)
        avg_acc = validate(epoch, args, deepsc)

        if avg_acc < record_acc:
            if not os.path.exists(args.checkpoint_path):
                os.makedirs(args.checkpoint_path)
            with open(args.checkpoint_path + '/checkpoint_{}.pth'.format(str(epoch + 1).zfill(2)), 'wb') as f:
                torch.save(deepsc.state_dict(), f)
            record_acc = avg_acc
    record_loss = []


    

        


