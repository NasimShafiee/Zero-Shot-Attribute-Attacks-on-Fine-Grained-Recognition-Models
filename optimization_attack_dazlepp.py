import argparse
from attack_core.optimization_attack_dazle import universal_end2end_attack

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

if dataset =="CUB": 
    dataloader = CUBDataLoaderImg(NFS_path_AoA,device,is_unsupervised_attr=False,is_balance=True)
elif dataset =="AWA2":
    dataloader = AWA2DataLoaderImg(NFS_path_AoA,device,is_unsupervised_attr=False,is_balance=True)
elif dataset =="SUN":
    dataloader = SUNDataLoaderImg(NFS_path_AoA,device,is_unsupervised_attr=False,is_balance=True)
model = Resnet_DAZLE(dataloader, torch.device("cuda:0"), dataset)
model.eval()

confidences  = [0.5, 2, 4, 8]
lambdas  =[1]
lambdas_2=[0]
lambdas_3=[0]
lambdas_4    = [2**-7, 2**-5, 2**-3, 0.5, 2]
lambdas_norm = [2**-7, 2**-5, 2**-3, 0.5, 2]

lambda_distance =0
data_indices= [3]
iteration=0
attention = False
partition = False 
num_clusters_list=[0]
min_acc = 100 
w_as ="function" # "function" \ "fc" \"variable"
scale_loss =False
log_folder="./log/"

# if dataset=="CUB":
#     confidences     = [8]
#     lambdas         = [4]
#     lambdas_4       = [4]
#     lambdas_norm    = [2]#[2]
#     lambdas_3       = [0]
#     num_clusters_list    = [0]
# elif dataset=="AWA2":
#     confidences     = [2]
#     lambdas         = [16]#[32]#[2,8,16,32]#[16]
#     lambdas_4       = [4]
#     lambdas_norm    = [0.5]#[4]
#     lambdas_3       = [0]
#     num_clusters_list    = [0]
# elif dataset=="SUN":
#     confidences     = [4]
#     lambdas         = [32]
#     lambdas_4       = [0.5]
#     lambdas_norm    = [0.5]#[3]
#     lambdas_3       = [0]
#     num_clusters_list    = [0]

for data_index in data_indices:

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

        for num_clusters in num_clusters_list:
            for confidence in reversed(confidences):
                for lam_norm in lambdas_norm:
                    for lam_2 in lambdas_2:
                        for lam_3 in lambdas_3:
                            for lam_4 in lambdas_4:
                                for lam in lambdas:
                                    text_tensorboard = "validationSet_scaleloss"+str(int(scale_loss))+"_attn"+str(int(attention))+"_"+str(num_clusters)+"clusters"+"_partitionEa09_"+str(int(partition))+"_"+dataset+"_bias_"+ w_as +"_losspow2_conf"+str(confidence)+"_lamNorm_"+str(lam_norm)+"_lam_"+str(lam)+"_"+str(lam_2)+"_"+str(lam_3)+"_"+str(lam_4)+str(lambda_distance)+"_data_"+str(data_index)
                                    attack_instance = universal_end2end_attack(dataloader.att.size(1),dataloader.att.size(0),[3,224,224], dataloader.w2v_att, dataloader.w2v_att.size(1),dataset, w_as,\
                                                                            device,\
                                                                            0.001, attention, partition, num_clusters, scale_loss, log_folder, text_tensorboard, if_dropout=True)

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
                                        attack_instance(model,data, data_validation, dataloader.att, dataloader.w2v_att, 50, confidence, num_epochs=500,\
                                                        norm_coefficient=lam_norm, loss_coefficient=lam, reg_e_coefficient=lam_2, reg_w_coefficient=lam_3, reg_utility=lam_4, reg_dist=lambda_distance,\
                                                        normalized_perturbation=False, project_perturbation=True,checkpoint_path="saved_universals_optimization/temp/"+text_tensorboard+".pt")

                                        # del data
                                        # torch.cuda.empty_cache()
                                        torch.save(attack_instance.state_dict(), "saved_universals_optimization/"+text_tensorboard+".pt")
                                    else:
                                        print("************************* loading previousely created universals *************************")
                                        attack_instance.load_state_dict(torch.load("saved_universals_optimization/"+text_tensorboard+".pt"))

                                    torch.save(attack_instance.e_a,"saved_universals_optimization/"+"nnW_utilityReg_nonUniformAttn_dazle++_attribute_universals"+text_tensorboard+".pt")
                                    torch.save(attack_instance.w_a_c,"saved_universals_optimization/"+"nnW_utilityReg_nonUniformAttn_dazle++_universals_weight"+text_tensorboard+".pt")
                                    
                                    # Report of Attribute Universals
                                    print("\t1) Report of Universals")
                                    attack_instance.utility_over_w_a_c(dataloader.w2v_att, dataloader.att)
                                    attack_instance.correlation_of_attribute_perturbations(dataloader)  
                                    attack_instance.correlation_of_class_perturbations(dataloader.w2v_att, dataloader.att, dataloader)
                                    attack_instance.major_attribute_universals()
                                    attack_instance.major_class_universals(dataloader.w2v_att, dataloader.att, k=20)
                                    attack_instance.major_weights_of_universals(dataloader.w2v_att, dataloader.att, k=100)
                                    attack_instance.graphs_class_attribute_relation(dataloader, dataloader.w2v_att, dataloader.att, k=5)
                                    
                                    # Validation
                                    acc_val = []
                                    norm_2_range = torch.arange(1.5,4.5,0.5)
                                    for i in norm_2_range:
                                        print("\t\tL2 norm scale: ",i.item())
                                        acc_s,_=attack_instance.attack_with_universals(model, data_validation, dataloader.att, dataloader.w2v_att, 100, norm_perturbation=2, norm_scale=i,attack_scenario="seen",normalized_perturbation=False, project_perturbation=True, validation=True)
                                        acc_val.append(acc_s.cpu())
                                        
                                    fig_1=plt.figure()
                                    ax = fig_1.add_subplot(111)
                                    ax.set_ylim(0,1.1)
                                    ax.set_xlabel("L2 norm")
                                    ax.set_ylabel("Accuracy")
                                    ax.set_title("confidence_"+str(confidence)+"lambda_"+str(lam))

                                    plt.plot(norm_2_range.cpu(), acc_val,'green')
                                    for i,j in zip(norm_2_range.cpu(), acc_val):
                                        ax.annotate("{:.2f}".format(j.item()),xy=(i,j-0.08))

                                    attack_instance.logger.add_figure("validation_accuracy", fig_1, iteration)
                                    attack_instance.logger.add_scalar("validation accuracy sum", sum(acc_val))

                                    # Test
                                    acc_seen = []
                                    acc_unseen = []
                                    attack_instance.eval()
                                    norm_2_range = torch.arange(1.5,4.5,0.5)
                                    print("\t2) Attack with Universals")
                                    for i in norm_2_range:
                                        print("\t\tL2 norm scale: ",i.item())
                                        acc_s,_=attack_instance.attack_with_universals(model, dataloader.data['test_seen'], dataloader.att, dataloader.w2v_att, 100, norm_perturbation=2, norm_scale=i,attack_scenario="seen",normalized_perturbation=False, project_perturbation=True)
                                        acc_u,_=attack_instance.attack_with_universals(model, dataloader.data['test_unseen'], dataloader.att, dataloader.w2v_att, 100, norm_perturbation=2, norm_scale=i,attack_scenario="unseen",normalized_perturbation=False, project_perturbation=True)
                                        acc_seen.append(acc_s.cpu())
                                        acc_unseen.append(acc_u.cpu())
                                        
                                    fig_1=plt.figure()
                                    ax = fig_1.add_subplot(111)
                                    ax.set_ylim(0,1.1)
                                    ax.set_xlabel("L2 norm")
                                    ax.set_ylabel("Accuracy")
                                    ax.set_title("confidence_"+str(confidence)+"lambda_"+str(lam))
                                    plt.plot(norm_2_range.cpu(), acc_seen,'green')
                                    for i,j in zip(norm_2_range.cpu(), acc_seen):
                                        ax.annotate("{:.2f}".format(j.item()),xy=(i,j-0.08))
                                    plt.plot(norm_2_range.cpu(), acc_unseen,'red')
                                    for i,j in zip(norm_2_range.cpu(), acc_unseen):
                                        ax.annotate("{:.2f}".format(j.item()),xy=(i,j+0.06))
                                    attack_instance.logger.add_figure("total_accuracy", fig_1, iteration)

                                    # Find the best configuration
                                    total_acc_seen   = sum(acc_seen)
                                    total_acc_unseen = sum(acc_unseen)
                                    total_acc_area = 0.5*(total_acc_seen + total_acc_unseen)
                                    attack_instance.logger.add_scalar("area under accuracy curve for hyperparam tuning", total_acc_area)
                                    if total_acc_area < min_acc:
                                        min_acc = total_acc_area
                                        print("\n\n@@@ best configuration until now @@@\t", text_tensorboard, "\tarea ",min_acc)
                                    iteration+=1

time.sleep(10)
