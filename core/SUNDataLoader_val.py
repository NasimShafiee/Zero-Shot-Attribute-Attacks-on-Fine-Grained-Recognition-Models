#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 12:11:40 2019

@author: war-machince
"""

import os,sys
#import scipy.io as sio
import torch
import numpy as np
import h5py
import time
import pickle
import pdb
from sklearn import preprocessing
from core.helper_func import mix_up
from global_setting import NFS_path_AoA
#%%
import scipy.io as sio
import pandas as pd
#%%
# import pdb
#%% These are for visualization
img_dir = os.path.join(NFS_path_AoA,'data/SUN/')
mat_path = os.path.join(NFS_path_AoA,'data/xlsa17/data/SUN/res101.mat')
attr_path = './attribute/SUN/new_des.csv'
#%%


class SUNDataLoader():
    def __init__(self, data_path, device, is_scale = False,is_unsupervised_attr = False,is_balance=True,hdf5_template=None):

        print(data_path)
        sys.path.append(data_path)

        self.data_path = data_path
        self.device = device
        self.dataset = 'SUN'
        print('$'*30)
        print(self.dataset)
        print('$'*30)
        self.datadir = self.data_path + 'data/{}/'.format(self.dataset)
        
        if hdf5_template is None:
            hdf5_template = 'feature_map_ResNet_101_{}.hdf5'
        
        self.path= self.datadir + hdf5_template.format(self.dataset)
        
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.is_scale = is_scale
        self.is_balance = is_balance
        if self.is_balance:
            print('Balance dataloader')
        self.is_unsupervised_attr = is_unsupervised_attr
        self.read_matdataset()
        self.get_idx_classes()
        self.I = torch.eye(self.allclasses.size(0)).to(device)

    def next_batch(self, batch_size):
        if self.is_balance:
            idx = []
            n_samples_class = max(batch_size //self.ntrain_class,1)
            sampled_idx_c = np.random.choice(np.arange(self.ntrain_class),min(self.ntrain_class,batch_size),replace=False).tolist()
            for i_c in sampled_idx_c:
                idxs = self.idxs_list[i_c]
                idx.append(np.random.choice(idxs,n_samples_class))
            idx = np.concatenate(idx)
            idx = torch.from_numpy(idx)
        else:
            idx = torch.randperm(self.ntrain)[0:batch_size]
    
        batch_feature = self.data['train_seen']['resnet_features'][idx].to(self.device)
        batch_label =  self.data['train_seen']['labels'][idx].to(self.device)
        batch_att = self.att[batch_label].to(self.device)
        return batch_label, batch_feature, batch_att
    
    def get_idx_classes(self):
        n_classes = self.seenclasses.size(0)
        self.idxs_list = []
        train_label = self.data['train_seen']['labels']
        for i in range(n_classes):
            idx_c = torch.nonzero(train_label == self.seenclasses[i].cpu()).cpu().numpy()
            idx_c = np.squeeze(idx_c)
            self.idxs_list.append(idx_c)
        return self.idxs_list
    
#    def next_batch_mix_up(self,batch_size):
#        Y1,S1,_=self.next_batch(batch_size)
#        Y2,S2,_=self.next_batch(batch_size)
#        S,Y=mix_up(S1,S2,Y1,Y2)
#        return Y,S,None
        
    def read_matdataset(self):

        print('_____')
        print(self.path)
        tic = time.perf_counter()
        hf = h5py.File(self.path, 'r')
        features = np.array(hf.get('feature_map'))
#        shape = features.shape
#        features = features.reshape(shape[0],shape[1],shape[2]*shape[3])
        labels = np.array(hf.get('labels'))
        trainval_loc = np.array(hf.get('trainval_loc'))
#        train_loc = np.array(hf.get('train_loc')) #--> train_feature = TRAIN SEEN
#        val_unseen_loc = np.array(hf.get('val_unseen_loc')) #--> test_unseen_feature = TEST UNSEEN
        test_seen_loc = np.array(hf.get('test_seen_loc'))
        test_unseen_loc = np.array(hf.get('test_unseen_loc'))
        
        if self.is_unsupervised_attr:
            print('Unsupervised Attr')
            class_path = './w2v/{}_class.pkl'.format(self.dataset)
            with open(class_path,'rb') as f:
                w2v_class = pickle.load(f)
            assert w2v_class.shape == (50,300)
            w2v_class = torch.tensor(w2v_class).float()
            
            U, s, V = torch.svd(w2v_class)
            reconstruct = torch.mm(torch.mm(U,torch.diag(s)),torch.transpose(V,1,0))
            print('sanity check: {}'.format(torch.norm(reconstruct-w2v_class).item()))
            
            print('shape U:{} V:{}'.format(U.size(),V.size()))
            print('s: {}'.format(s))
            
            self.w2v_att = torch.transpose(V,1,0).to(self.device)
            self.att = torch.mm(U,torch.diag(s)).to(self.device)
            self.normalize_att = torch.mm(U,torch.diag(s)).to(self.device)
            
        else:
            print('Expert Attr')
            att = np.array(hf.get('att'))
            self.att = torch.from_numpy(att).float().to(self.device)
            
            original_att = np.array(hf.get('original_att'))
            self.original_att = torch.from_numpy(original_att).float().to(self.device)
            
            w2v_att = np.array(hf.get('w2v_att'))
            self.w2v_att = torch.from_numpy(w2v_att).float().to(self.device)
            
            self.normalize_att = self.original_att/100
        
        print('Finish loading data in ',time.perf_counter()-tic)
        
        train_feature = features[trainval_loc]
        test_seen_feature = features[test_seen_loc]
        test_unseen_feature = features[test_unseen_loc]
        if self.is_scale:
            scaler = preprocessing.MinMaxScaler()
    
            train_feature = scaler.fit_transform(train_feature)
            test_seen_feature = scaler.fit_transform(test_seen_feature)
            test_unseen_feature = scaler.fit_transform(test_unseen_feature)

        train_feature = torch.from_numpy(train_feature).float() #.to(self.device)
        test_seen_feature = torch.from_numpy(test_seen_feature) #.float().to(self.device)
        test_unseen_feature = torch.from_numpy(test_unseen_feature) #.float().to(self.device)

        train_label = torch.from_numpy(labels[trainval_loc]).long() #.to(self.device)
        test_unseen_label = torch.from_numpy(labels[test_unseen_loc]) #.long().to(self.device)
        test_seen_label = torch.from_numpy(labels[test_seen_loc]) #.long().to(self.device)

        self.seenclasses = torch.from_numpy(np.unique(train_label.cpu().numpy())).to(self.device)
        self.unseenclasses = torch.from_numpy(np.unique(test_unseen_label.cpu().numpy())).to(self.device)
        self.ntrain = train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()

#        self.train_mapped_label = map_label(train_label, self.seenclasses)

        self.data = {}
        self.data['train_seen'] = {}
        self.data['train_seen']['resnet_features'] = train_feature
        self.data['train_seen']['labels']= train_label


        self.data['train_unseen'] = {}
        self.data['train_unseen']['resnet_features'] = None
        self.data['train_unseen']['labels'] = None

        self.data['test_seen'] = {}
        self.data['test_seen']['resnet_features'] = test_seen_feature
        self.data['test_seen']['labels'] = test_seen_label

        self.data['test_unseen'] = {}
        self.data['test_unseen']['resnet_features'] = test_unseen_feature
        self.data['test_unseen']['labels'] = test_unseen_label

# ------------------------------------------------------------
# ------------------------------------------------------------
# ------------------------------------------------------------
# ------------------------------------------------------------

class SUNDataLoaderImg():
    def __init__(self, data_path, device, is_scale = False,is_unsupervised_attr = False,is_balance=True,hdf5_template=None):

        print(data_path)
        sys.path.append(data_path)

        self.data_path = data_path
        self.device = device
        self.dataset = 'SUN'
        print('$'*30)
        print(self.dataset)
        print('$'*30)
        self.datadir = self.data_path + 'data/{}/'.format(self.dataset)
        
        if hdf5_template is None:
            hdf5_template = 'img_feature_map_ResNet_101_{}.hdf5'
        
        self.path= self.datadir + hdf5_template.format(self.dataset)
        
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.is_scale = is_scale
        self.is_balance = is_balance
        if self.is_balance:
            print('Balance dataloader')
        self.is_unsupervised_attr = is_unsupervised_attr
        self.read_matdataset()
        self.get_idx_classes()
        self.I = torch.eye(self.allclasses.size(0)).to(device)

    def next_batch(self, batch_size):
        if self.is_balance:
            idx = []
            n_samples_class = max(batch_size //self.ntrain_class,1)
            sampled_idx_c = np.random.choice(np.arange(self.ntrain_class),min(self.ntrain_class,batch_size),replace=False).tolist()
            for i_c in sampled_idx_c:
                idxs = self.idxs_list[i_c]
                idx.append(np.random.choice(idxs,n_samples_class))
            idx = np.concatenate(idx)
            idx = torch.from_numpy(idx)
        else:
            idx = torch.randperm(self.ntrain)[0:batch_size]
    
        batch_feature = self.data['train_seen']['resnet_features'][idx].to(self.device)
        batch_image   = self.data['train_seen']['images'][idx].to(self.device)
        batch_label   =  self.data['train_seen']['labels'][idx].to(self.device)
        batch_att     = self.att[batch_label].to(self.device)
        return batch_label, batch_feature, batch_image, batch_att
    
    def get_idx_classes(self):
        n_classes = self.seenclasses.size(0)
        self.idxs_list = []
        train_label = self.data['train_seen']['labels']
        for i in range(n_classes):
            idx_c = torch.nonzero(train_label == self.seenclasses[i].cpu()).cpu().numpy()
            idx_c = np.squeeze(idx_c)
            self.idxs_list.append(idx_c)
        return self.idxs_list
    
#    def next_batch_mix_up(self,batch_size):
#        Y1,S1,_=self.next_batch(batch_size)
#        Y2,S2,_=self.next_batch(batch_size)
#        S,Y=mix_up(S1,S2,Y1,Y2)
#        return Y,S,None
        
    def read_matdataset(self):

        print('_____')
        print(self.path)
        tic = time.perf_counter()
        hf = h5py.File(self.path, 'r')
        features = np.array(hf.get('feature_map'))
        images   = np.array(hf.get('images'))
#        shape = features.shape
#        features = features.reshape(shape[0],shape[1],shape[2]*shape[3])
        labels = np.array(hf.get('labels'))
        trainval_loc = np.array(hf.get('trainval_loc'))
        train_loc = np.array(hf.get('train_loc')) #--> train_feature = TRAIN SEEN
        val_unseen_loc = np.array(hf.get('val_unseen_loc')) #--> test_unseen_feature = TEST UNSEEN
        test_seen_loc = np.array(hf.get('test_seen_loc'))
        test_unseen_loc = np.array(hf.get('test_unseen_loc'))
        
        if self.is_unsupervised_attr:
            print('Unsupervised Attr')
            class_path = './w2v/{}_class.pkl'.format(self.dataset)
            with open(class_path,'rb') as f:
                w2v_class = pickle.load(f)
            assert w2v_class.shape == (50,300)
            w2v_class = torch.tensor(w2v_class).float()
            
            U, s, V = torch.svd(w2v_class)
            reconstruct = torch.mm(torch.mm(U,torch.diag(s)),torch.transpose(V,1,0))
            print('sanity check: {}'.format(torch.norm(reconstruct-w2v_class).item()))
            
            print('shape U:{} V:{}'.format(U.size(),V.size()))
            print('s: {}'.format(s))
            
            self.w2v_att = torch.transpose(V,1,0).to(self.device)
            self.att = torch.mm(U,torch.diag(s)).to(self.device)
            self.normalize_att = torch.mm(U,torch.diag(s)).to(self.device)
            
        else:
            print('Expert Attr')
            att = np.array(hf.get('att'))
            self.att = torch.from_numpy(att).float().to(self.device)
            
            original_att = np.array(hf.get('original_att'))
            self.original_att = torch.from_numpy(original_att).float().to(self.device)
            
            w2v_att = np.array(hf.get('w2v_att'))
            self.w2v_att = torch.from_numpy(w2v_att).float().to(self.device)
            
            self.normalize_att = self.original_att/100
        
        print('Finish loading data in ',time.perf_counter()-tic)
        
        train_feature = features[trainval_loc]
        test_seen_feature = features[test_seen_loc]
        test_unseen_feature = features[test_unseen_loc]
        train_only_feature  = features[train_loc]
        valid_only_feature  = features[val_unseen_loc]

        train_image = images[trainval_loc]
        test_seen_image = images[test_seen_loc]
        test_unseen_image = images[test_unseen_loc]
        train_only_image  = images[train_loc]
        valid_only_image  = images[val_unseen_loc]

        if self.is_scale:
            scaler = preprocessing.MinMaxScaler()
    
            train_feature = scaler.fit_transform(train_feature)
            test_seen_feature = scaler.fit_transform(test_seen_feature)
            test_unseen_feature = scaler.fit_transform(test_unseen_feature)
            train_only_feature  = scaler.fit_transform(train_only_feature)
            valid_only_feature  = scaler.fit_transform(valid_only_feature)

        train_feature = torch.from_numpy(train_feature).float() #.to(self.device)
        test_seen_feature = torch.from_numpy(test_seen_feature) #.float().to(self.device)
        test_unseen_feature = torch.from_numpy(test_unseen_feature) #.float().to(self.device)
        train_only_feature  = torch.from_numpy(train_only_feature)
        valid_only_feature  = torch.from_numpy(valid_only_feature)

        train_image = torch.from_numpy(train_image).float() #.to(self.device)
        test_seen_image = torch.from_numpy(test_seen_image) #.float().to(self.device)
        test_unseen_image = torch.from_numpy(test_unseen_image) #.float().to(self.device)
        train_only_image  = torch.from_numpy(train_only_image)
        valid_only_image  = torch.from_numpy(valid_only_image)

        train_label = torch.from_numpy(labels[trainval_loc]).long() #.to(self.device)
        test_unseen_label = torch.from_numpy(labels[test_unseen_loc]) #.long().to(self.device)
        test_seen_label = torch.from_numpy(labels[test_seen_loc]) #.long().to(self.device)
        train_only_label = torch.from_numpy(labels[train_loc]).long()
        valid_only_label = torch.from_numpy(labels[val_unseen_loc]).long()

        self.seenclasses = torch.from_numpy(np.unique(train_label.cpu().numpy())).to(self.device)
        self.unseenclasses = torch.from_numpy(np.unique(test_unseen_label.cpu().numpy())).to(self.device)
        self.ntrain = train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()

#        self.train_mapped_label = map_label(train_label, self.seenclasses)

        self.data = {}
        self.data['train_seen'] = {}
        self.data['train_seen']['resnet_features'] = train_feature
        self.data['train_seen']['images'] = train_image
        self.data['train_seen']['labels'] = train_label


        self.data['train_unseen'] = {}
        self.data['train_unseen']['resnet_features'] = None
        self.data['train_unseen']['images'] = None
        self.data['train_unseen']['labels'] = None

        self.data['test_seen'] = {}
        self.data['test_seen']['resnet_features'] = test_seen_feature
        self.data['test_seen']['images'] = test_seen_image
        self.data['test_seen']['labels'] = test_seen_label

        self.data['test_unseen'] = {}
        self.data['test_unseen']['resnet_features'] = test_unseen_feature
        self.data['test_unseen']['images'] = test_unseen_image
        self.data['test_unseen']['labels'] = test_unseen_label
        
        self.data['train_only_seen'] = {}
        self.data['train_only_seen']['resnet_features'] = train_only_feature
        self.data['train_only_seen']['images'] = train_only_image
        self.data['train_only_seen']['labels'] = train_only_label

        self.data['valid_only_seen'] = {}
        self.data['valid_only_seen']['resnet_features'] = valid_only_feature
        self.data['valid_only_seen']['images'] = valid_only_image
        self.data['valid_only_seen']['labels'] = valid_only_label