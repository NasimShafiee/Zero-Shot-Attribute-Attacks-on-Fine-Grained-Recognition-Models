import argparse
from attack_core.optimization_attack_dazle_test import universal_end2end_attack
import torchvision
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

p_norm = 2

from attack_core.model_with_resnet import Resnet_DAZLE
from torch.utils.tensorboard import SummaryWriter
import time

sys.path.insert(1, '/mnt/raptor/nasim/adversarial-fine-grained-on-nips/neurIPS20_CompositionZSL')
from global_setting import NFS_path_AoA,save_NFS
from core.CUBDataLoader_val import CUBDataLoader, CUBDataLoaderImg
from core.AWA2DataLoader_val import AWA2DataLoader, AWA2DataLoaderImg
from core.SUNDataLoader_val import SUNDataLoader, SUNDataLoaderImg
NFS_path_AoA = NFS_path_AoA
import matplotlib.pyplot as plt
#%%
idx_GPU = 0
device = torch.device("cuda:{}".format(idx_GPU) if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed_all(0)
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

#%%
#dataloader = CUBDataLoaderImg(NFS_path_AoA,device,is_unsupervised_attr=False,is_balance=True)
if dataset =="CUB": 
    dataloader = CUBDataLoaderImg(NFS_path_AoA,device,is_unsupervised_attr=False,is_balance=True)
elif dataset =="AWA2":
    dataloader = AWA2DataLoaderImg(NFS_path_AoA,device,is_unsupervised_attr=False,is_balance=True)
elif dataset =="SUN":
    dataloader = SUNDataLoaderImg(NFS_path_AoA,device,is_unsupervised_attr=False,is_balance=True)

model = Resnet_DAZLE(dataloader, torch.device("cuda:0"), dataset)
model.eval()

confidences= [2]#[2]#[0.5,1,2,,4,8,16]
lambdas  =[2]#[0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
lambdas_2=[0]#[0, 0.0001, 0.001, 0.01, 0.1]
lambdas_3=[0]#[0.001, 0.1, 1, 4, 8, 16, 32, 64, 128]
lambdas_4=[1]#[0.1]#[0.5, 0.1, 0.01, 0.001]
lambdas_norm =[0.5]#[0,0.5,1,1.5,2]

lambda_distance =0#0.001#0.001
data_indices= [3]#[7000]#[3]#[5000]#[3,5,10]
iteration=0
attention = False#False
partition = False #True#False
num_clusters_list=[0]#,5,10,20,30,40,50,60,100,200] #5,10,20
min_acc = 100 # infinite
w_as ="function" # "function" \ "fc" \"variable"
scale_loss =False
WE_reg ="" # "WEReg" | ""
text_tensorboard = "_TrueCheckpoint"+dataset+"_test_new_transferability_CEZSL_"+w_as+"W_"+WE_reg+"_(dazle++configWith"+w_as+"W)_onDAZLE_certain_hyperparamtuned"
log_folder ="./log_transferability_datasets_second/"

# Load the best result of hyperparam tuning for each dataset
if w_as=="function":#and WE_reg=="":
    if dataset =="CUB": 
        # text_tensorboard_DAZLE = dataset+"_bias_"+ w_as +"_losspow2_conf8_lam_4_0_0_4_data_3"
        # text_tensorboard_DAZLE = "_scaleLoss0_attn0_0clusters_partitionEa09_0_CUB_bias_function_losspow2_conf8_lamNorm_2_lam_4_0_0_40_data_3"
        # text_tensorboard_DAZLE = "_scaleLoss0_attn0_0clusters_partitionEa09_0_CUB_bias_function_losspow2_conf2_lamNorm_0.5_lam_4_0_0_20_data_3"
        text_tensorboard_model = "_L2_CUB_conf8_lamNorm_-3.0_lam_1_1.0"
    elif dataset =="AWA2":
        # text_tensorboard_DAZLE = dataset+"_bias_"+ w_as +"_losspow2_conf4_lam_16_0_0_16_data_3"
        # text_tensorboard_DAZLE = "_scaleLoss0_attn0_0clusters_partitionEa09_0_AWA2_bias_function_losspow2_conf2_lamNorm_4_lam_16_0_0_40_data_3"
        # text_tensorboard_DAZLE = "_scaleLoss0_attn0_0clusters_partitionEa09_0_AWA2_bias_function_losspow2_conf2_lamNorm_0.5_lam_4_0_0_80_data_3"
        text_tensorboard_model = "_L2_AWA2_conf8_lamNorm_-3.0_lam_1_1.0"
    # elif dataset =="APY":
    #     # text_tensorboard_DAZLE = dataset+"_bias_"+ w_as +"_losspow2_conf16_lam_16_0_0_1_data_3"
    #     text_tensorboard_DAZLE = dataset+"_bias_"+ w_as +"_losspow2_conf16_lam_16_0_0_1_data_3"
    elif dataset =="SUN":
        # text_tensorboard_DAZLE = dataset+"_bias_"+ w_as +"_losspow2_conf16_lam_32_0_0_0.5_data_3"
        # text_tensorboard_DAZLE = "_scaleLoss0_attn0_0clusters_partitionEa09_0_SUN_bias_function_losspow2_conf4_lamNorm_0.5_lam_32_0_0_0.50_data_3"
        # text_tensorboard_DAZLE = "_scaleLoss0_attn0_0clusters_partitionEa09_0_SUN_bias_function_losspow2_conf2_lamNorm_0.5_lam_16_0_0_0.50_data_3"
        text_tensorboard_model = "_L2_SUN_conf8_lamNorm_-1.0_lam_1_-7.0"
# if dataset=="CUB":
#     confidences     = [8]
#     lambdas         = [4]
#     lambdas_4       = [4]
#     lambdas_norm    = [2]#[2]
#     lambdas_3       = [0]
#     num_clusters_list    = [0]
# elif dataset=="AWA2":
#     confidences     = [2]
#     lambdas         = [16]
#     lambdas_4       = [4]
#     lambdas_norm    = [0]#[4]
#     lambdas_3       = [0]
#     num_clusters_list    = [0]
# elif dataset=="SUN":
#     confidences     = [4]
#     lambdas         = [32]
#     lambdas_4       = [0.5]
#     lambdas_norm    = [0]#[3]
#     lambdas_3       = [0]
#     num_clusters_list    = [0]

if dataset=="CUB":
    confidence  = 8
    lam         = 1
    lam_4       = 2**1
    lam_norm    = 2**-3
    lam_3       = 0
    num_clusters    = 0
elif dataset=="AWA2":
    confidence  = 8
    lam         = 1
    lam_4       = 2**1
    lam_norm    = 2**-3
    lam_3       = 0
    num_clusters    = 0
elif dataset=="SUN":
    confidence  = 8
    lam         = 1
    lam_4       = 2**-7
    lam_norm    = 2**-1
    lam_3       = 0
    num_clusters    = 0

attack_instance = universal_end2end_attack(dataloader.att.size(1),dataloader.att.size(0),[3,224,224], dataloader.w2v_att, dataloader.w2v_att.size(1),dataset, w_as,\
                                        device,\
                                        0.001, attention, partition, num_clusters, scale_loss, log_folder, text_tensorboard, if_dropout=True)


# create indices
len_dataloader       = dataloader.data['train_seen']['labels'].size(0)-2 
len_dataloader_train = dataloader.data['train_only_seen']['labels'].size(0)-2 
len_dataloader_valid = dataloader.data['valid_only_seen']['labels'].size(0)-2
# read indices
with open("/mnt/raptor/nasim/adversarial-fine-grained-on-nips/neurIPS20_CompositionZSL/trainInidicesData_val_"+dataset+".txt", "r") as file:
    train_indices = torch.tensor( eval(file.readline()) )
with open("/mnt/raptor/nasim/adversarial-fine-grained-on-nips/neurIPS20_CompositionZSL/validInidicesData_val_"+dataset+".txt", "r") as file:
    valid_indices = torch.tensor( eval(file.readline()) )

data={  "resnet_features":dataloader.data['train_only_seen']['resnet_features'][train_indices],\
        "images":dataloader.data['train_only_seen']['images'][train_indices],\
        "labels":dataloader.data['train_only_seen']['labels'][train_indices]}
data_validation ={  "resnet_features":dataloader.data['valid_only_seen']['resnet_features'][valid_indices],\
        "images":dataloader.data['valid_only_seen']['images'][valid_indices],\
        "labels":dataloader.data['valid_only_seen']['labels'][valid_indices]}

if w_as == "function":
    attack_instance.e_a   = torch.load("/mnt/raptor/nasim/others/CE-GZSL/saved_universals_optimization/"+"nnW_utilityReg_nonUniformAttn_dazle++_attribute_universals"+text_tensorboard_model+".pt")
    attack_instance.w_a_c = torch.load("/mnt/raptor/nasim/others/CE-GZSL/saved_universals_optimization/"+"nnW_utilityReg_nonUniformAttn_dazle++_universals_weight"+text_tensorboard_model+".pt")

    img_grid = torchvision.utils.make_grid(attack_instance.e_a)
    w_grid   = torchvision.utils.make_grid(attack_instance.w_a_c)

    attack_instance.logger.add_image('universals per attribute', img_grid)
    attack_instance.logger.add_image('weights per attribute', w_grid)

# torch.save(attack_instance.e_a,"saved_universals_optimization/"+"nnW_utilityReg_nonUniformAttn_dazle++_attribute_universals"+text_tensorboard+".pt")
# torch.save(attack_instance.w_a_c,"saved_universals_optimization/"+"nnW_utilityReg_nonUniformAttn_dazle++_universals_weight"+text_tensorboard+".pt")



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
attack_instance.logger.add_figure("total_accuracy", fig_1)

time.sleep(10)
