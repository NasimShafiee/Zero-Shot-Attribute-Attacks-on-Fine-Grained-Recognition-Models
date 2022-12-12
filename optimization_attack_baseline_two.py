import argparse
from attack_core.optimization_attack_baseline_two import universal_end2end_attack

import torch
import numpy as np
import os,sys
import random 
pwd = os.getcwd()

print(pwd)
parent = '/'.join(pwd.split('/')[:])
sys.path.insert(0,parent)
os.chdir(parent)
print(parent)

parser = argparse.ArgumentParser()
parser.add_argument("-d","--dataset", default="CUB", choices=["AWA2","CUB","APY","SUN"])
parser.add_argument("-p","--norm", default="L2", choices=["L2","Linf"])
parser.add_argument("--lamnorm",   default="1")
parser.add_argument("--confidence", default="8")
args = parser.parse_args()
dataset = args.dataset
print(" ---->>>> RUN FOR DATASET ",dataset)

from attack_core.model_with_resnet import Resnet_DAZLE
from torch.utils.tensorboard import SummaryWriter
import time

from global_setting import NFS_path_AoA,save_NFS
from core.CUBDataLoader_val import CUBDataLoader, CUBDataLoaderImg
from core.AWA2DataLoader_val import AWA2DataLoader, AWA2DataLoaderImg
from core.SUNDataLoader_val import SUNDataLoader, SUNDataLoaderImg
import matplotlib.pyplot as plt
#%%
idx_GPU = 0
device = torch.device("cuda:{}".format(idx_GPU) if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


p_norm =2 if args.norm=="L2" else torch.tensor(float('inf')) #2 or torch.tensor(float('inf'))

print(args.confidence,args.lamnorm)
confidence  = float(args.confidence)#[0.5, 2, 4, 8]
lam  =1
lam_norm = 2**float(args.lamnorm)#[2**-7, 2**-5, 2**-3, 0.5, 2]

data_index= 3
min_acc = 100 
scale_loss =False
log_folder="./log__Uniform_L2/" if p_norm==2 else "./log__Uniform_SudotLinf/"
txt_config = dataset+"_conf"+str(int(confidence))+"_lamNorm_"+str(np.log2(lam_norm))+"_lam_"+str(lam)
text_tensorboard = "Uniform_L2_"+txt_config if p_norm==2 else "Uniform_SudotLinf_"+txt_config
if not os.path.isfile("saved_universals_optimization/"+text_tensorboard+".pt"):
        
    if dataset =="CUB": 
        dataloader = CUBDataLoaderImg(NFS_path_AoA,device,is_unsupervised_attr=False,is_balance=True)
    elif dataset =="AWA2":
        dataloader = AWA2DataLoaderImg(NFS_path_AoA,device,is_unsupervised_attr=False,is_balance=True)
    elif dataset =="SUN":
        dataloader = SUNDataLoaderImg(NFS_path_AoA,device,is_unsupervised_attr=False,is_balance=True)


    model = Resnet_DAZLE(dataloader, torch.device("cuda:0"), dataset)
    model.eval()

    attack_instance = universal_end2end_attack(dataloader.att.size(1),dataloader.att.size(0),[3,224,224],dataloader.w2v_att.size(1),dataset, "uniform",\
                                            device,\
                                            0.001, log_folder, text_tensorboard)

        # create indices
    len_dataloader       = dataloader.data['train_seen']['labels'].size(0)-2 
    len_dataloader_train = dataloader.data['train_only_seen']['labels'].size(0)-2 
    len_dataloader_valid = dataloader.data['valid_only_seen']['labels'].size(0)-2

    # train_indices  = torch.randperm(len_dataloader_train)[0:int(len_dataloader*(0.8)/3)]
    # valid_indices  = torch.randperm(len_dataloader_valid)[0:int(len_dataloader*(0.2)/3)] 

    # save indices
    # with open("./data_indices/trainInidicesData_val_"+dataset+".txt", "w") as file:
    #     file.write(str(train_indices.tolist()))
    # with open("./data_indices/validInidicesData_val_"+dataset+".txt", "w") as file:
    #     file.write(str(valid_indices.tolist()))

    # read indices
    with open("./data_indices/trainInidicesData_val_"+dataset+".txt", "r") as file:
        train_indices = torch.tensor( eval(file.readline()) )
    with open("./data_indices/validInidicesData_val_"+dataset+".txt", "r") as file:
        valid_indices = torch.tensor( eval(file.readline()) )

    data={  "resnet_features":dataloader.data['train_only_seen']['resnet_features'][train_indices],\
            "images":dataloader.data['train_only_seen']['images'][train_indices],\
            "labels":dataloader.data['train_only_seen']['labels'][train_indices]}
    data_validation ={  "resnet_features":dataloader.data['valid_only_seen']['resnet_features'][valid_indices],\
            "images":dataloader.data['valid_only_seen']['images'][valid_indices],\
            "labels":dataloader.data['valid_only_seen']['labels'][valid_indices]}

    # Check if there is universal created before or not
    if not os.path.isfile("saved_universals_optimization/"+text_tensorboard+".pt"):
        print("************************* creating universals *************************")
        attack_instance.train()
        attack_instance(model,data, data_validation, dataloader.att,\
                        dataloader.w2v_att,50,confidence, num_epochs=500,\
                        loss_norm = p_norm, loss_coefficient=lam,\
                        norm_coefficient=lam_norm, normalized_perturbation=False, project_perturbation=True,\
                        checkpoint_path="saved_universals_optimization/temp/"+text_tensorboard+".pt")                     
        torch.save(attack_instance.state_dict(), "saved_universals_optimization/"+text_tensorboard+".pt")
    else:

        print("************************* loading previousely created universals *************************")
        attack_instance.load_state_dict(torch.load("saved_universals_optimization/"+text_tensorboard+".pt"))



    torch.save(attack_instance.state_dict(), "saved_universals_optimization/"+text_tensorboard+".pt")
    attack_instance.load_state_dict(torch.load("saved_universals_optimization/"+text_tensorboard+".pt"))


# Validation
    acc_val = []
    if p_norm ==2:
        norm__range = torch.arange(2,14,4)
        p_norm = torch.tensor(p_norm)
    else:
        norm__range  = torch.arange(0.02,0.14,0.04)
    for i in norm__range:
        print("\t\tValidation L norm: ",i.item())
        acc_s,_=attack_instance.attack_with_universals(model, data_validation, dataloader.att, dataloader.w2v_att, 100, norm_perturbation=i, norm_p=p_norm,attack_scenario="seen",normalized_perturbation=False, project_perturbation=True, validation=True) 
        acc_val.append(acc_s.cpu())

    fig_1=plt.figure()
    ax = fig_1.add_subplot(111)
    ax.set_ylim(0,1.1)
    ax.set_xlabel("L norm")
    ax.set_ylabel("Accuracy")
    ax.set_title("confidence_"+str(confidence)+"lambda_"+str(lam))

    plt.plot(norm__range.cpu(), acc_val,'green')
    for i,j in zip(norm__range.cpu(), acc_val):
        ax.annotate("{:.2f}".format(j.item()),xy=(i,j-0.08))

    attack_instance.logger.add_figure("validation_accuracy", fig_1)
    attack_instance.logger.add_scalar("validation accuracy sum", sum(acc_val))

    # Test
    acc_seen = []
    acc_unseen = []
    attack_instance.eval()

    if p_norm ==2:
        norm__range = torch.arange(2,14,4)
        p_norm = torch.tensor(p_norm)
    else:
        norm__range  = torch.arange(0.02,0.14,0.04)
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