import argparse
from attack_core.optimization_attack_baseline_random import universal_end2end_attack


import torch
import os,sys
import random
import numpy as np

pwd = os.getcwd()
parent = '/'.join(pwd.split('/')[:])
sys.path.insert(0,parent)
os.chdir(parent)


parser = argparse.ArgumentParser()
parser.add_argument("-d","--dataset", default="CUB", choices=["AWA2","CUB","APY","SUN"])
parser.add_argument("-p","--norm", default="L2", choices=["L2","Linf"])
args = parser.parse_args()
dataset = args.dataset
print(" ---->>>> RUN FOR DATASET ",dataset)

from attack_core.model_with_resnet import Resnet_DAZLE

from global_setting import NFS_path_AoA,save_NFS
from core.CUBDataLoader_val import CUBDataLoader, CUBDataLoaderImg
from core.AWA2DataLoader_val import AWA2DataLoader, AWA2DataLoaderImg
from core.SUNDataLoader_val import SUNDataLoader, SUNDataLoaderImg
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
import time

idx_GPU = 0
device = torch.device("cuda:{}".format(idx_GPU) if torch.cuda.is_available() else "cpu")


torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


p_norm =2 if args.norm=="L2" else torch.tensor(float('inf')) #2 or torch.tensor(float('inf'))
if dataset =="CUB": 
    dataloader = CUBDataLoaderImg(NFS_path_AoA,device,is_unsupervised_attr=False,is_balance=True)
elif dataset =="AWA2":
    dataloader = AWA2DataLoaderImg(NFS_path_AoA,device,is_unsupervised_attr=False,is_balance=True)
elif dataset =="SUN":
    dataloader = SUNDataLoaderImg(NFS_path_AoA,device,is_unsupervised_attr=False,is_balance=True)

model = Resnet_DAZLE(dataloader, torch.device("cuda:0"), dataset)
model.eval()

lam = 4
confidence = 2
data_index = 3
log_folder = "./log_baselines/"
text_tensorboard = dataset+"_baseline_three_random_universal"
text_tensorboard = text_tensorboard+"_L2" if p_norm==2 else text_tensorboard+"_SudotLinf"


attack_instance = universal_end2end_attack(dataloader.att.size(1),dataloader.att.size(0),[3,224,224],dataloader.w2v_att.size(1), dataset, "variable",\
                                           device,\
                                           0.001, log_folder, text_tensorboard)

# create indices
len_dataloader       = dataloader.data['train_seen']['labels'].size(0)-2 
len_dataloader_train = dataloader.data['train_only_seen']['labels'].size(0)-2 
len_dataloader_valid = dataloader.data['valid_only_seen']['labels'].size(0)-2



acc_seen = []
acc_unseen = []
attack_instance.eval()

if p_norm ==2:
    norm__range = torch.arange(0,15,1)
    p_norm = torch.tensor(p_norm)
else:
    norm__range  = torch.arange(0.0,0.15,0.01)
print("\t2) Attack with Universals")
for i in norm__range:
    print("\t\tL norm: ",i.item())
    acc_s,_=attack_instance.attack_with_universals(model, dataloader.data['test_seen'], dataloader.att, dataloader.w2v_att, 100, norm_perturbation=i, norm_p=p_norm,attack_scenario="seen",normalized_perturbation=False, project_perturbation=True)
    acc_u,_=attack_instance.attack_with_universals(model, dataloader.data['test_unseen'], dataloader.att, dataloader.w2v_att, 100, norm_perturbation=i, norm_p=p_norm,attack_scenario="unseen",normalized_perturbation=False, project_perturbation=True)
    acc_seen.append(acc_s.cpu())
    acc_unseen.append(acc_u.cpu())


# write on tensorboard

fig_1=plt.figure()
ax = fig_1.add_subplot(111)
ax.set_ylim(0,1.1)
ax.set_xlabel("L norm")
ax.set_ylabel("Accuracy")
ax.set_title("confidence_"+str(confidence)+"lambda_"+str(lam))
plt.plot(norm__range.cpu(), acc_seen,'green')
for i,j in zip(norm__range.cpu(), acc_seen):
    ax.annotate("{:.2f}".format(j.item()),xy=(i,j-0.08))
plt.plot(norm__range.cpu(), acc_unseen,'red')
for i,j in zip(norm__range.cpu(), acc_unseen):
    ax.annotate("{:.2f}".format(j.item()),xy=(i,j+0.06))
    
attack_instance.logger.add_figure('total_accuracy', fig_1)

time.sleep(10)
