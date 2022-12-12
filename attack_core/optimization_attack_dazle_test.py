from typing_extensions import final
from torch._C import device
import tqdm
import torch
from torch.functional import norm
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchray.attribution.guided_backprop import GuidedBackpropContext
import torch.nn.functional as F
import skimage.transform
import skimage
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import time
import pandas as pd
import numpy as np
import torch.nn as nn
import scipy
import random
import sys
from global_setting import NFS_path_AoA,save_NFS
sys.path.insert(1,'../early_stopping/early-stopping-pytorch')
from pytorchtools import EarlyStopping

# import wandb
random.seed(0)

# Transform and Inverse Transform
input_size = 224
transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(input_size),
        torchvision.transforms.CenterCrop(input_size),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

inverse_transform = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=[0.,0.,0.], std=[1/0.229, 1/0.224, 1/0.225]),\
            torchvision.transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.,1.,1.])
])

min_t = transform(torch.zeros(1,3,224,224)).min()
max_t = transform(torch.ones(1,3,224,224)).max()
temperature =0.005 # DEfault0.005

# Universal Generation and Attack
class universal_end2end_attack(torch.nn.Module):
    def __init__(self, num_attributes, num_classes, image_size, w2v_vector, w2v_size, dataset, w_as="function",\
                                        device=torch.device("cpu"), learning_rate=0.001, attention=False, partition=False, num_clusters=0, scale_loss=False, log_folder="./log_1/", log_title="_", if_dropout=False):

        super(universal_end2end_attack, self).__init__()
        print("w_as ################### ",w_as)
        self.num_attributes = num_attributes
        self.num_classes    = num_classes
        self.attention      = attention 
        self.partition      = partition
        self.num_clusters   = num_clusters
        self.w_as           = w_as
        self.scale_loss     = scale_loss
        self.dataset        = dataset

        # Finetune w2v semantic vector
        #self.w2v_vector     =  torch.nn.Parameter(F.normalize(torch.tensor(w2v_vector.detach().clone(), device=device)))

        if self.w_as=="function":
            self.w_a_c      = torch.nn.Parameter(torch.zeros(num_attributes,\
                                                            w2v_size, device= device))
        elif self.w_as=="function1":
            self.w_a_c      = torch.nn.Parameter(torch.zeros(num_attributes,\
                                                            w2v_size, device= device))
            self.N1         = torch.nn.Sequential(  nn.Linear(num_attributes, num_attributes, device=device),
                                                    nn.LeakyReLU()
                                                    )
            self.N2         = torch.nn.Sequential(  nn.Linear(w2v_size, w2v_size, device=device),
                                                    nn.LeakyReLU()
                                                    )
        elif self.w_as=="function2":
            self.w_a_c      = torch.nn.Parameter(torch.zeros(num_attributes,\
                                                            w2v_size, device= device))
            self.b_a_c      = torch.nn.Parameter(torch.zeros(num_attributes,\
                                                            1, device= device))
        elif self.w_as=="fc":
            self.w_a_c      = torch.nn.Sequential(  nn.Linear(1+w2v_size, 40, device=device),
                                                    nn.LeakyReLU(),
                                                    nn.Linear(40, 10, device=device),
                                                    nn.LeakyReLU(),
                                                    nn.Linear(10, 1, device=device),
                                                    nn.Tanh()
                                                    )

        else:#as variable
            self.w_a_c      = torch.nn.Parameter(torch.zeros(num_attributes,\
                                                         num_classes, device= device))
       
        self.e_a        = torch.nn.Parameter(torch.zeros(num_attributes,\
                                            image_size[0], image_size[1],\
                                            image_size[2], device=device))

        if num_clusters>0:
            self.e_a        = torch.nn.Parameter(torch.zeros(num_clusters,\
                                                image_size[0], image_size[1],\
                                                image_size[2], device=device))
            self.clusters   = pd.read_csv("./data/clusters/"+dataset+"_attribute_"+str(num_clusters)+"clusters.csv")

        if self.partition:
            num_partitions = 10/9#10/8.5#10/9
            self.partitions_e_a = torch.zeros_like(self.e_a).to(device)
            image_size     = 3*224*224
            attr_univ_size = int(image_size//num_partitions)
            print("each partition percentage size", attr_univ_size/image_size,attr_univ_size)
            for i in range(self.e_a.size(0)):
                indices = torch.randperm(image_size)[0:attr_univ_size]
                (self.partitions_e_a[i]).reshape(-1)[indices] = 1

        # self.dropout        = torch.nn.Dropout3d(p=0.2)
        self.dropout        = torch.nn.Dropout2d(p=0.2)
        self.dropout_w      = torch.nn.Dropout(p=0.2)
        self.dropout_label  = if_dropout
       
        # self.optimizer  = torch.optim.RMSprop([self.w_a_c, self.e_a], lr=learning_rate)
        # self.optimizer  = torch.optim.SGD([self.w_a_c, self.e_a], lr=learning_rate, momentum=0.9)
        if self.w_as=="fc":
            self.optimizer  = torch.optim.Adam(list(self.w_a_c.parameters())+[self.e_a], lr=learning_rate)
            self.scheduler  = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9, verbose=True)
        elif self.w_as=="function":    
            self.optimizer  = torch.optim.Adam([self.w_a_c, self.e_a], lr=learning_rate)
            self.scheduler  = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9, verbose=True)
        elif self.w_as=="function1":    
            self.optimizer  = torch.optim.Adam(list(self.N1.parameters())+list(self.N2.parameters())+[self.w_a_c, self.e_a], lr=learning_rate)
            self.scheduler  = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9, verbose=True)
        elif self.w_as=="function2":    
            self.optimizer  = torch.optim.Adam([self.w_a_c, self.b_a_c, self.e_a], lr=learning_rate)
            self.scheduler  = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9, verbose=True)
       
        self.device     = device
       
        self.logger     = SummaryWriter(log_folder+log_title)

        if dataset=="CUB":
            self.attr_names = pd.read_csv(NFS_path_AoA+'data/CUB/CUB_200_2011/attributes.txt', delimiter=" ", header=None, names=["num","name"]).drop(axis=1,columns="num")
            self.class_names= pd.read_csv(NFS_path_AoA+'data/xlsa17/data/CUB/allclasses.txt',    delimiter=".", header=None, names=["num","name"]).drop(axis=1,columns="num")
        elif dataset=="AWA2":
            self.attr_names = pd.read_csv(NFS_path_AoA+'data/AWA2/Animals_with_Attributes2/predicates.txt', delimiter="\t", header=None, names=["num","name"]).drop(axis=1,columns="num")
            self.class_names= pd.read_csv(NFS_path_AoA+'data/AWA2/Animals_with_Attributes2/classes.txt',    delimiter="\t", header=None, names=["num","name"]).drop(axis=1,columns="num")
        elif dataset=="APY":
            self.attr_names = pd.read_csv(NFS_path_AoA+'data/APY/attribute_data/attribute_names.txt', header=None, names=["name"])
            self.class_names= pd.read_csv(NFS_path_AoA+'data/APY/attribute_data/class_names.txt', header=None, names=["name"])
        elif dataset=="SUN":
            mat             = scipy.io.loadmat(NFS_path_AoA+'data/SUN/attributes.mat')
            attr_names      = [mat["attributes"][:,0][i][0] for i in range(len(mat["attributes"]))]
            self.attr_names = pd.DataFrame(attr_names,columns=["name"])

            mat=scipy.io.loadmat(NFS_path_AoA+"data/xlsa17/data/SUN/att_splits.mat")
            class_names =[mat["allclasses_names"][:,0][i][0] for i in range(len(mat["allclasses_names"]))]
            self.class_names = pd.DataFrame(class_names,columns=["name"])

        if self.w_as!="fc":
            torch.nn.init.uniform_(self.w_a_c, a=0.5,     b=0.5)
        torch.nn.init.uniform_(self.e_a,   a=0.00001, b=0.001)
   
    def class_sampler(self, dataloader, batch_size):
        label_set        = set(dataloader['labels'].tolist())
        label_sample     = random.sample(label_set,min(batch_size,len(label_set)))
        if len(label_set)<batch_size:
            num_sam_perclass = int(batch_size//len(label_set)) +1
        else:
            num_sam_perclass =1
        # print("@@@@@@@@@@@@@@@@@@@@ num samples\t",num_sam_perclass) 
        samples_indices  = [random.sample((dataloader['labels']==val).nonzero(as_tuple=True)[0].tolist(), num_sam_perclass) for i, val in enumerate(label_sample)]
        
        if num_sam_perclass!=1:
            samples_indices = [item for sublist in samples_indices  for item in sublist]
            samples_indices = [random.choice(samples_indices) for _ in range(batch_size)]

        # print(samples_indices)
        label_data = dataloader['labels'][samples_indices].squeeze().to(self.device)
        image_data = dataloader['images'][samples_indices].reshape(-1,3,224,224).to(self.device)


        return label_data, image_data

   
    def attribute_score(self, model, batch_image, attention=False):    
        pred        = model(batch_image)
        pred        = torch.argmax(pred, 1)
        attr_score  = model.dazle.package_out['A_p'] * model.dazle.package_out['S_p']
        if attention == True:
            size0       = batch_image.size(0)
            attn        = (model.dazle.package_out['A']).reshape(size0, self.num_attributes, 7, 7)
            upsam_func  = torch.nn.Upsample(size=(224, 224))
            attn = upsam_func(attn).unsqueeze(-3)
            return attr_score, pred, attn
        else:
            return attr_score, pred
   
   
   
    def compute_weights(self, w2v_vector, semantic_vector):
        if self.w_as=="function":
            # Calculate Weights of perturbations combination
            w_comb              = torch.einsum('aw,aw,ya->ay', w2v_vector, self.w_a_c, semantic_vector) 
            w_comb_normalized   = torch.tanh(w_comb)
            # softmax = torch.nn.Softmax(dim=0)
            # w_comb_normalized   = softmax(w_comb)
            # w_comb_normalized   = torch.nn.functional.leaky_relu(w_comb)

        elif self.w_as=="function1":
            w_comb              = torch.einsum('aw,aw,ya->ay', self.N2(w2v_vector), self.w_a_c, self.N1(semantic_vector)) 
            w_comb_normalized   = torch.tanh(w_comb)

        elif self.w_as=="function2":
            # Calculate Weights of perturbations combination
            w_comb              = torch.einsum('aw,aw,ya->ay', w2v_vector, self.w_a_c, semantic_vector)  + self.b_a_c
            w_comb_normalized   = torch.tanh(w_comb)

        elif self.w_as=="fc":
            out = []
            num_classes = semantic_vector.size(0)
            for c in range(num_classes):
                z_c = semantic_vector[c].reshape(-1,1)
                # print(w2v_vector.size(),z_c.size())
                input = torch.cat([w2v_vector, z_c], dim=-1)
                # print(input.size())
                output=self.w_a_c(input)
                out.append(output)
            w_comb_normalized = torch.stack(out).reshape(-1,num_classes)
            # print("w_comb_normalized\n\n",w_comb_normalized.size(),"\n\n",w_comb_normalized)
           
        else:
            w_comb              = self.w_a_c
            w_comb_normalized   = torch.tanh(w_comb)
            # inv_norm    = torch.norm(w_comb, p=2, dim=0).pow(-1)
            # w_comb_normalized = torch.einsum("y,ay->ay", inv_norm, w_comb)
           
        return w_comb_normalized
   

   
    def log_function(self, iter, loss, loss_1, loss_2):
        self.logger.add_scalar('Loss/train', loss, iter)
        self.logger.add_scalar('Loss1/train', loss_1, iter)
        self.logger.add_scalar('Loss2/train', loss_2, iter)
        # wandb.log({"batch loss norm of perturbation":loss_1.item()})
        # wandb.log({"batch loss attack strength":loss_2.item()})
        # wandb.log({"batch loss":loss.item()})
       
        # img_grid = torchvision.utils.make_grid(self.e_a)
        # images   = wandb.Image(img_grid, caption="attribute universals")
        # w_grid   = torchvision.utils.make_grid(self.w_a_c)
        # weights  = wandb.Image(w_grid, caption="combination weights")
        # wandb.log({"universals": images})
        # wandb.log({"weights": weights})
       
        # mean_norm = torch.norm(self.e_a.reshape(self.e_a.size(0),-1), p=2, dim=-1).mean()
        # wandb.log({"average L2 universals":mean_norm})
       
        if iter%50==0:
            img_grid = torchvision.utils.make_grid(self.e_a)
            self.logger.add_image('universals per attribute', img_grid, iter)
            # self.logger.add_image('weights per attribute', w_grid,iter)
            # self.logger.add_image('universals per attribute', img_grid, iter)
            # self.logger.add_image('weights per attribute', w_grid, iter)


   
    def loss_adversraial_condition(self, final_adv_scores, corr_batch_label, confidence):
        # Find first max but not the correct label (y_i!=j)
        other_class_max_score, other_class = torch.topk(final_adv_scores, 2, dim=-1, largest=True)            
        second_max_index = torch.nonzero(other_class != corr_batch_label)
        list = []
        list.append(second_max_index[0])
        for i in range(1,second_max_index.size(0)):
            if second_max_index[i-1,0]!=second_max_index[i,0]:
                list.append(second_max_index[i])
        second_max_index = torch.stack(list)

        second_max_class = other_class[second_max_index[:,0],second_max_index[:,1]]
        second_max_score = other_class_max_score[second_max_index[:,0],second_max_index[:,1]]
       
        final_adv_index  = torch.cat([torch.arange(corr_batch_label.size(0), device=self.device).unsqueeze(1),\
                                    corr_batch_label], dim=1)
       
        final_adv_scores = final_adv_scores[final_adv_index[:,0], final_adv_index[:,1]]

        S_yi     = final_adv_scores
        max_S_j  = second_max_score
        loss     = torch.einsum("b,b->b", ((S_yi - max_S_j + confidence) > 0),\
                                           (S_yi - max_S_j + confidence)).sum()
       
        return loss



    def calculate_perturbation(self, w_comb, corr_batch_attn, corr_batch_label, train=True):
        
        if self.num_clusters>0:
            e_a = self.e_a[self.clusters.attr_cluster]
        else:
            e_a = self.e_a

        # Normalized e_a
        # e_a = F.normalize(e_a,dim=0)
        
        if train and self.dropout_label:
            # e = torch.einsum('ay,achw->ychw', w_comb, self.dropout(self.e_a.unsqueeze(0)).squeeze()) 3d dropout
            if not self.attention:
                e = torch.einsum('ay,achw->ychw', w_comb, self.dropout(e_a))
                perturbation    = e[corr_batch_label].reshape(-1,3,224,224)
            else:
                e = torch.einsum('bhw, ba, achw->bchw', corr_batch_attn.reshape(-1,self.num_attributes,224,224).sum(1).reshape(-1,224,224).detach().clone(), w_comb.permute(1,0)[corr_batch_label].reshape(-1,self.num_attributes), self.dropout(e_a))
                perturbation    = e.reshape(-1,3,224,224)
        else:
            if not self.attention:
                e = torch.einsum('ay,achw->ychw', w_comb, e_a)
                perturbation    = e[corr_batch_label].reshape(-1,3,224,224)
            else:
                e = torch.einsum('bhw, ba, achw->bchw', corr_batch_attn.reshape(-1,self.num_attributes,224,224).sum(1).reshape(-1,224,224).detach().clone(), w_comb.permute(1,0)[corr_batch_label].reshape(-1,self.num_attributes), e_a)
                perturbation    = e.reshape(-1,3,224,224)
        
        # Normalize perturbation
        # normalized_perturbation = 2 * F.normalize(perturbation,dim=0)
                 
        return perturbation



    def calculate_partial_perturbation(self, w_comb, corr_batch_label, train=True):

        # Normalized e_a
        # e_a = F.normalize(e_a,dim=0)

        if self.num_clusters>0:
            e_a = self.e_a[self.clusters.attr_cluster]
            partitions_e_a = self.partitions_e_a[self.clusters.attr_cluster]
        else:
            e_a = self.e_a
            partitions_e_a = self.partitions_e_a

        if train and self.dropout_label:
            e = torch.einsum('ay,achw->ychw', w_comb, self.dropout(e_a)*partitions_e_a.detach().clone())
            perturbation    = e[corr_batch_label].reshape(-1,3,224,224)
        else:
            e = torch.einsum('ay,achw->ychw', w_comb, e_a*partitions_e_a.detach().clone())
            perturbation    = e[corr_batch_label].reshape(-1,3,224,224)

        # Normalize perturbation
        # normalized_perturbation = 2 * F.normalize(perturbation,dim=0)
                 
        return perturbation


    def add_perturbation(self, normalized_batch_image, perturbation, normalized_perturbation=True, project_perturbation=False, scale=True, norm_p=1, norm_perturbation=1):
        if normalized_perturbation:
            
            if scale:# Scale Perturbation
                perturbation = transform(self.scale_perturbation(inverse_transform(perturbation), norm_p, norm_perturbation))

            if project_perturbation:
                adv_batch_image = torch.clamp(normalized_batch_image + perturbation,\
                                              min=min_t, max=max_t)
            else:
                adv_batch_image = normalized_batch_image + perturbation
               
        else:
            if scale:# Scale Perturbation
                perturbation = self.scale_perturbation(perturbation, norm_p, norm_perturbation)

            batch_image     = inverse_transform(normalized_batch_image)
            if project_perturbation:
                adv_batch_image = transform(torch.clamp(batch_image + perturbation,\
                                            min=0, max=1))
            else:
                adv_batch_image = transform(batch_image + perturbation)

        return adv_batch_image

    
    
    def scale_perturbation(self, perturbation, norm_p, perturbation_scale):
        
        if norm_p==2:
            # Magnitude of Perturbations
            norm_p_img   = torch.norm(perturbation,p=norm_p,dim=(1,2,3)).mean().to(self.device)

            # Scaling Perturbations
            scale_l = torch.tensor(perturbation_scale, requires_grad=False).to(self.device)
            ratio_l = norm_p_img/scale_l
            perturbation_scaled = perturbation/ratio_l
        
        else:
            norm_p_img = torch.norm(perturbation.reshape(-1,3,224,224),p=float('inf'),dim=(2,3)).reshape(-1,3,1,1).to(self.device)
            scale_l = torch.tensor(perturbation_scale,requires_grad=False).to(self.device)
            ratio_l = norm_p_img/scale_l
            perturbation_scaled = perturbation/ratio_l

       

        return perturbation_scaled



    def image_visualization(self, images, perturbations, adv_images, adv_title, cln_title, attack_scenario, norm_perturbation, norm_p, validation=False):
        fig, axs = plt.subplots(9, np.ceil(images.size(0)/3).astype(int),figsize=(90,50))
        if np.ceil(images.size(0)/3).astype(int)>1:
            for i in range(images.size(0)):
                axs[0+3*(i%3),i//3].imshow(images[i].squeeze().detach().clone().cpu().permute(1,2,0).numpy())
                axs[1+3*(i%3),i//3].imshow(perturbations[i].squeeze().detach().clone().cpu().permute(1,2,0).numpy())
                axs[2+3*(i%3),i//3].imshow(adv_images[i].squeeze().detach().clone().cpu().permute(1,2,0).numpy())
                axs[0+3*(i%3),i//3].set_title(cln_title[i])
                axs[2+3*(i%3),i//3].set_title(adv_title[i])
                axs[0+3*(i%3),i//3].axes.xaxis.set_visible(False)
                axs[0+3*(i%3),i//3].axes.yaxis.set_visible(False)
                axs[1+3*(i%3),i//3].axes.xaxis.set_visible(False)
                axs[1+3*(i%3),i//3].axes.yaxis.set_visible(False)
                axs[2+3*(i%3),i//3].axes.xaxis.set_visible(False)
                axs[2+3*(i%3),i//3].axes.yaxis.set_visible(False)
        else:
            for i in range(images.size(0)):
                axs[0+3*(i%3)].imshow(images[i].squeeze().detach().clone().cpu().permute(1,2,0).numpy())
                axs[1+3*(i%3)].imshow(perturbations[i].squeeze().detach().clone().cpu().permute(1,2,0).numpy())
                axs[2+3*(i%3)].imshow(adv_images[i].squeeze().detach().clone().cpu().permute(1,2,0).numpy())
                axs[0+3*(i%3)].set_title(cln_title[i])
                axs[2+3*(i%3)].set_title(adv_title[i])
                axs[0+3*(i%3)].axes.xaxis.set_visible(False)
                axs[0+3*(i%3)].axes.yaxis.set_visible(False)
                axs[1+3*(i%3)].axes.xaxis.set_visible(False)
                axs[1+3*(i%3)].axes.yaxis.set_visible(False)
                axs[2+3*(i%3)].axes.xaxis.set_visible(False)
                axs[2+3*(i%3)].axes.yaxis.set_visible(False)
        fig.tight_layout()
        self.logger.add_figure('attack/'+attack_scenario+'_L'+str(norm_p.item())+"="+str(norm_perturbation.item()), fig, int(not(validation)))
        
        

    ####################################################################
    ###              Train the Universals                            ###
    ####################################################################

    def forward(self, model, dataloader, dataloader_val, semantic_vector, w2v_vector, batch_size, \
                confidence, num_epochs=10, loss_norm=2, norm_coefficient=1, loss_coefficient=10, \
                reg_e_coefficient=0.001,reg_w_coefficient=0.001, reg_utility=0.001, reg_dist=0, \
                normalized_perturbation=True, project_perturbation=False, checkpoint_path="./"):
        # Initialize Variables
        dataset_length      = len(dataloader['images'])
        dataset_length_val  = len(dataloader_val['images'])
        iter=0
        global temperature
        early_stopping = EarlyStopping(patience=20, verbose=True, path=checkpoint_path)#20
       
        # Iterate over Epochs
        for epoch in range(num_epochs):
            # Change Temperature
            # temperature -=0.0001

            loss_train  =[]
            loss_train_1=[]
            loss_train_2=[]
            dist_train=[]
            #Iterate over batch
            for i in range(0, dataset_length, batch_size):            

                self.optimizer.zero_grad()
                
                batch_label   = dataloader['labels'][i:i+batch_size].to(self.device)
                batch_image   = dataloader['images'][i:i+batch_size].to(self.device)

                # batch_label, batch_image = self.class_sampler(dataloader, batch_size)

                # Calculate weights for combining universals
                w_comb = self.compute_weights(w2v_vector, semantic_vector)
                # if self.dropout_label:
                #     w_comb = self.dropout_w(self.compute_weights(w2v_vector, semantic_vector))
                # else:
                #     w_comb = self.compute_weights(w2v_vector, semantic_vector)

                # Calculate attribute scores
                attribute_score, c, attn_map = self.attribute_score(model, batch_image, attention=True)
                # attribute_score, c, attn_map = self.attribute_score(model, batch_image, attention=True)
               
                # Filter only the correct samples in the batch
                corr_batch_image = batch_image[torch.nonzero(c==batch_label)]#.detach().clone()
                corr_batch_label = batch_label[torch.nonzero(c==batch_label)]#.detach().clone()
                corr_batch_attn  = attn_map[torch.nonzero(c==batch_label)]#.detach().clone()
                corr_attr_score  = attribute_score[torch.nonzero(c==batch_label)].squeeze()

                # No Correct Prediction To Attack
                if (c==batch_label).sum()<=0:
                    continue

                # Calculate Adversarial Perturbations Per Class
                if not self.partition:
                    perturbation    = self.calculate_perturbation(w_comb, corr_batch_attn, corr_batch_label, train=True)
                    # perturbation    = self.scale_perturbation(perturbation, norm_p=2, perturbation_scale=2)
                else:
                    perturbation    = self.calculate_partial_perturbation(w_comb, corr_batch_label, train=True)
                    # perturbation    = self.scale_perturbation(perturbation, norm_p=2, perturbation_scale=2)

                # Create Perturbations for the batch
                if not self.scale_loss:
                    adv_batch_image = self.add_perturbation(corr_batch_image.reshape(-1,3,224,224),\
                                                            perturbation,\
                                                            normalized_perturbation=normalized_perturbation,\
                                                            project_perturbation=project_perturbation, scale=False)
                else:
                    adv_batch_image = self.add_perturbation(corr_batch_image.reshape(-1,3,224,224),\
                                                            perturbation,\
                                                            normalized_perturbation=normalized_perturbation,\
                                                            project_perturbation=project_perturbation, scale=True, norm_p=1, norm_perturbation=100)

                # Attack
                adv_attribute_score, adv_c  = self.attribute_score(model, adv_batch_image.reshape(-1,3,224,224))
                final_adv_scores            = torch.einsum('ba,ya->by', adv_attribute_score, semantic_vector)


                # Distance Regularization 
                if reg_dist>0:
                    sample_size=min(2,corr_batch_image.size(0))
                    idx_dist = torch.randperm(corr_batch_image.size(0))[0:sample_size]
                    sample_e_a_size = min(30, self.e_a.size(0))
                    idx_e_a  = torch.randperm(self.e_a.size(0))[:sample_e_a_size]
                    reg_batch_image = self.add_perturbation(corr_batch_image[idx_dist].reshape(-1,3,224,224).repeat(sample_e_a_size,1,1,1),\
                                                            self.e_a[idx_e_a].reshape(-1,3,224,224).repeat(sample_size,1,1,1),\
                                                            normalized_perturbation=normalized_perturbation,\
                                                            project_perturbation=project_perturbation, scale=False)
                    # Attack
                    reg_attribute_score, reg_c  = self.attribute_score(model, reg_batch_image.reshape(-1,3,224,224))
                    # print("reg attr score",reg_attribute_score.size())
                    distance =0 
                    if self.num_clusters>0:
                        for _,i in enumerate(idx_e_a):
                            idx_t  = self.clusters.attr_idx[self.clusters.attr_cluster==i].tolist()
                            idx_nt = self.clusters.attr_idx[self.clusters.attr_cluster!=i].tolist()
                            distance += ( corr_attr_score[idx_dist].reshape(-1, self.num_attributes)[:, idx_nt] - reg_attribute_score.reshape(sample_size, sample_e_a_size,-1)[:,j,idx_nt] ).pow(2).sum(dim=-1).sqrt().mean()
                    else:
                        for j,i in enumerate(idx_e_a):
                            idx_nt = torch.nonzero(torch.arange(312)!=i).tolist()
                            distance += ( corr_attr_score.reshape(-1, self.num_attributes)[idx_dist].reshape(-1, self.num_attributes)[:, idx_nt] - reg_attribute_score.reshape(sample_size, sample_e_a_size,-1)[:,j,idx_nt] ).pow(2).sum(dim=-1).sqrt().mean()
                else:
                    distance=torch.tensor([0], device=self.device)

                # Calculate Loss function
                if loss_norm==2:
                    loss_1  = 0.5 * torch.norm(self.e_a.reshape(self.e_a.size(0),-1), p=loss_norm, dim=-1).pow(loss_norm).sum()#/ (self.num_attributes)
                else:
                    # loss_1  = 0.5 * torch.norm(self.e_a.reshape(self.e_a.size(0),-1), p=loss_norm, dim=-1).sum()
                    t = temperature / 0.224 #(0.005) / 0.224
                    loss_1  = torch.einsum('achw,achw->achw', (self.e_a.abs() - t )>0 , self.e_a.abs() - t ).sum()

                loss_2  = self.loss_adversraial_condition(final_adv_scores, corr_batch_label, confidence)

                regularization_e = torch.norm(perturbation.reshape(perturbation.size(0),-1), p=1, dim=-1).sum()
                if self.w_as!="fc":
                    regularization_w = torch.norm(self.w_a_c.reshape(1,-1), p=2, dim=-1).pow(2).squeeze()
                else:
                    regularization_w=torch.zeros(1, device=self.device)

                norms = torch.norm(self.e_a.reshape(self.e_a.size(0),-1), p=2, dim=-1)
                regularization_std  = ((norms-norms.mean())**2).sum()
                utility             = w_comb.abs().sum(1).pow(2).squeeze()
                L2_utility          = torch.norm(utility, p=2)#  /(self.num_classes * self.num_attributes) 
                # print("train attr scores",corr_attr_score.size(), adv_attribute_score.size())
                # delta_attr_scores   = torch.einsum("ba, ba->ba",(corr_attr_score - adv_attribute_score),  (w_comb.permute(1,0)[corr_batch_label].reshape(-1,self.num_attributes).abs()<0.1).to(torch.float) )
                # loss_4 = delta_attr_scores.sum()
                # print(delta_attr_scores.size(), loss_4)
                # loss    = norm_coefficient*loss_1 + reg_dist * distance + loss_coefficient * loss_2 + reg_utility * L2_utility + reg_e_coefficient*regularization_e + reg_w_coefficient*regularization_w #+ 0.1*loss_4#+ reg_w_coefficient*regularization_std
                loss    = norm_coefficient*loss_1 + loss_coefficient * loss_2 + reg_utility * L2_utility 
                loss_train_1.append(loss_1.item())
                loss_train_2.append(loss_2.item())
                loss_train.append(loss.item())
                dist_train.append(distance.item())
                # Backpropagate
                loss.backward()
                clipping_value = 1 
                torch.nn.utils.clip_grad_norm_([self.w_a_c, self.e_a], clipping_value)

                self.optimizer.step()    

                # with torch.no_grad():
                #     if iter%100==0:
                #         attn_grid = torchvision.utils.make_grid(attn_map[0].unsqueeze(1))
                #         self.logger.add_image("attention map", attn_grid, int(i//batch_size))

                iter+=1
            
            # Log values
            self.log_function(epoch, sum(loss_train)/len(loss_train), sum(loss_train_1)/len(loss_train_1), sum(loss_train_2)/len(loss_train_2))
            self.logger.add_scalar('Distance/train', sum(dist_train)/len(dist_train), epoch)


            ## Validation step
            #Iterate over batch
            with torch.no_grad():
                loss_val =[]
                loss_val_1=[]
                loss_val_2=[]
                loss_val_3=[]
                dist_val=[]
                for i in range(0, dataset_length_val, batch_size):
                
                    batch_label   = dataloader_val['labels'][i:i+batch_size].to(self.device)
                    batch_image   = dataloader_val['images'][i:i+batch_size].to(self.device)

                    # Calculate weights for combining universals
                    w_comb = self.compute_weights(w2v_vector, semantic_vector)
                
                    # Calculate attribute scores
                    attribute_score, c, attn_map = self.attribute_score(model, batch_image, attention=True)
                
                    # Filter only the correct samples in the batch
                    corr_batch_image = batch_image[torch.nonzero(c==batch_label)]
                    corr_batch_label = batch_label[torch.nonzero(c==batch_label)]
                    corr_batch_attn  = attn_map[torch.nonzero(c==batch_label)]#.detach().clone()
                    corr_attr_score  = attribute_score[torch.nonzero(c==batch_label)].squeeze()

                    # No correct prediction to attack
                    if (c==batch_label).sum()<=0:
                        continue
                    
                    # Calculate Adversarial Perturbations Per Class
                    if not self.partition:
                        perturbation    = self.calculate_perturbation(w_comb, corr_batch_attn, corr_batch_label, train=False)
                        # perturbation    = self.scale_perturbation(perturbation, norm_p=2, perturbation_scale=2)
                    else:
                        perturbation    = self.calculate_partial_perturbation(w_comb, corr_batch_label, train=False)
                        # perturbation    = self.scale_perturbation(perturbation, norm_p=2, perturbation_scale=2)

                    # Create Perturbations for the batch
                    if not self.scale_loss:
                        adv_batch_image = self.add_perturbation(corr_batch_image.reshape(-1,3,224,224),\
                                                                perturbation,\
                                                                normalized_perturbation=normalized_perturbation,\
                                                                project_perturbation=project_perturbation, scale=False)
                    else:
                        adv_batch_image = self.add_perturbation(corr_batch_image.reshape(-1,3,224,224),\
                                                                perturbation,\
                                                                normalized_perturbation=normalized_perturbation,\
                                                                project_perturbation=project_perturbation, scale=True, norm_p=1, norm_perturbation=100)

                    # Attack
                    adv_attribute_score, adv_c  = self.attribute_score(model, adv_batch_image.reshape(-1,3,224,224))
                    final_adv_scores            = torch.einsum('ba,ya->by', adv_attribute_score, semantic_vector)
                

                    # Distance Regularization 
                    if reg_dist>0:
                        sample_size=min(2,corr_batch_image.size(0))
                        idx_dist = torch.randperm(corr_batch_image.size(0))[0:sample_size]
                        sample_e_a_size = min(30, self.e_a.size(0))
                        idx_e_a  = torch.randperm(self.e_a.size(0))[:sample_e_a_size]
                        reg_batch_image = self.add_perturbation(corr_batch_image[idx_dist].reshape(-1,3,224,224).repeat(sample_e_a_size,1,1,1),\
                                                                self.e_a[idx_e_a].reshape(-1,3,224,224).repeat(sample_size,1,1,1),\
                                                                normalized_perturbation=normalized_perturbation,\
                                                                project_perturbation=project_perturbation, scale=False)
                        # Attack
                        reg_attribute_score, reg_c  = self.attribute_score(model, reg_batch_image.reshape(-1,3,224,224))
                        # print("reg attr score",reg_attribute_score.size())
                        distance =0 
                        if self.num_clusters>0:
                            for j,i in enumerate(idx_e_a):
                                idx_t  = self.clusters.attr_idx[self.clusters.attr_cluster==i].tolist()
                                idx_nt = self.clusters.attr_idx[self.clusters.attr_cluster!=i].tolist()
                                distance += ( corr_attr_score[idx_dist].reshape(-1, self.num_attributes)[:, idx_nt] - reg_attribute_score.reshape(sample_size, sample_e_a_size,-1)[:,j,idx_nt] ).pow(2).sum(dim=-1).sqrt().mean()
                        else:
                            for _,i in enumerate(idx_e_a):
                                idx_nt = torch.nonzero(torch.arange(self.num_attributes)!=i).tolist()
                                distance += ( corr_attr_score.reshape(-1, self.num_attributes)[idx_dist].reshape(-1, self.num_attributes)[:, idx_nt] - reg_attribute_score.reshape(sample_size, sample_e_a_size,-1)[:,j,idx_nt] ).pow(2).sum(dim=-1).sqrt().mean()
                    else:
                        distance=torch.tensor([0],device=self.device)


                    # Calculate Loss function
                    if loss_norm==2:
                        loss_1  = 0.5 * torch.norm(self.e_a.reshape(self.e_a.size(0),-1), p=loss_norm, dim=-1).pow(loss_norm).sum()#/ (self.num_attributes)
                    else:
                        # Linf 
                        # loss_1  = 0.5 * torch.norm(self.e_a.reshape(self.e_a.size(0),-1), p=loss_norm, dim=-1).sum()
                        # SudoLinf 
                        t = temperature/0.224
                        loss_1  = torch.einsum('achw,achw->achw', (self.e_a.abs() - t )>0 , self.e_a.abs() - t ).sum()

                    # loss_1  = 0.5 * torch.norm(self.e_a.reshape(1,-1), p=loss_norm, dim=-1).pow(loss_norm).squeeze()#/ (self.num_attributes)
                    loss_2  = self.loss_adversraial_condition(final_adv_scores, corr_batch_label, confidence)
                    regularization_e = torch.norm(perturbation.reshape(perturbation.size(0),-1), p=1, dim=-1).sum()
                    if self.w_as!="fc":
                        regularization_w = torch.norm(self.w_a_c.reshape(1,-1), p=2, dim=-1).pow(2).squeeze()
                    else:
                        regularization_w=torch.zeros(1, device=self.device)
                    norms = torch.norm(self.e_a.reshape(self.e_a.size(0),-1), p=2, dim=-1)
                    regularization_std = ((norms-norms.mean())**2).sum()
                    utility             = w_comb.abs().sum(1).pow(2).squeeze()
                    L2_utility          = torch.norm(utility, p=2)#/(self.num_classes * self.num_attributes)
                    # print("validation attr scores",corr_attr_score.size(), adv_attribute_score.size())
                    # delta_attr_scores   = torch.einsum("ba, ba->ba",(corr_attr_score - adv_attribute_score),  (w_comb.permute(1,0)[corr_batch_label].reshape(-1,self.num_attributes).abs()<0.1).to(torch.float) )
                    # loss_4 = delta_attr_scores.sum()
                    # print(delta_attr_scores.size(), loss_4)
                    loss    = norm_coefficient*loss_1 + reg_dist * distance + loss_coefficient * loss_2 + reg_utility * L2_utility + reg_e_coefficient*regularization_e + reg_w_coefficient*regularization_w #+ 0.1*loss_4 #+ reg_w_coefficient*regularization_std
                    loss_val.append(loss.item())
                    loss_val_1.append(loss_1.item())
                    loss_val_2.append(loss_2.item())
                    loss_val_3.append(L2_utility.item())
                    dist_val.append(distance.item())

                self.logger.add_scalar('Loss1/validation', sum(loss_val_1)/len(loss_val_1), epoch)
                self.logger.add_scalar('Loss2/validation', sum(loss_val_2)/len(loss_val_2), epoch)
                self.logger.add_scalar('Loss3/validation', sum(loss_val_3)/len(loss_val_3), epoch)
                self.logger.add_scalar('Loss/validation', sum(loss_val)/len(loss_val), epoch)
                self.logger.add_scalar('Distance/validation', sum(dist_val)/len(dist_val), epoch)

                early_stopping( sum(loss_val)/len(loss_val), self)
                if early_stopping.early_stop:
                    print("Early stopping at epoch\t", epoch)
                    break
            # if  self.scheduler.get_last_lr()[0]>0.0005:
            #     self.scheduler.step()
        # load the last checkpoint with the best model
        self.load_state_dict(torch.load(checkpoint_path))
        return    


         

    ####################################################################
    ###              Attack with Universals                          ###
    ####################################################################
   
    def attack_with_universals(self, model, dataloader, semantic_vector, w2v_vector, \
                               batch_size, norm_perturbation=1, norm_p=1, \
                               attack_scenario="seen", normalized_perturbation=True, \
                               project_perturbation=False, validation=False):
        with torch.no_grad():
            length_dataset          = dataloader['resnet_features'].size(0)
            num_attacked_samples    = 0
            model_accuracy          = 0
            model_initial_accuracy  = 0
            list_diff_img = []

            # Calculate Adversarial Perturbations Per Class
            w_comb                  = self.compute_weights(w2v_vector, semantic_vector)
            if self.num_clusters>0:
                self.perturbation_per_class  = torch.einsum('ay,achw->ychw', w_comb, self.e_a[self.clusters.attr_cluster]) 
            else:
                self.perturbation_per_class  = torch.einsum('ay,achw->ychw', w_comb, self.e_a)
            pert_grid = torchvision.utils.make_grid(self.perturbation_per_class)
            self.logger.add_image("perturbation_per_class",pert_grid)

            img_grid = torchvision.utils.make_grid(self.e_a)
            self.logger.add_image('universals per attribute/test', img_grid)
           
            list_of_true_classes = []
            list_of_adv_classes  = []

            # Iterate over dataset
            for i in range(0,length_dataset,batch_size):
               
                batch_label   = dataloader['labels'][i:i+batch_size].to(self.device)
                batch_image   = dataloader['images'][i:i+batch_size].to(self.device)
                batch_att     = semantic_vector[batch_label].to(self.device)          
               
                # True predictions
                pred = model(batch_image)
                pred = torch.argmax(pred, 1)
                attr = model.dazle.package_out['A_p'] * model.dazle.package_out['S_p']
                if i==0:
                    cln_log_attn_1 = model.dazle.package_out['A']
                    cln_log_attn_2 = model.dazle.package_out['A_p']

                # If correctly predicted, we consider it to attack
                correctly_predicted = (pred == batch_label)
                batch_label        = batch_label[correctly_predicted]
                batch_image        = batch_image[correctly_predicted]
                batch_att          = semantic_vector[batch_label].to(self.device)
                pred               = pred[correctly_predicted]
                attr               = attr[correctly_predicted]
                # Wrong prediction for all batch
                if correctly_predicted.sum()<=0:
                    print("wrong prediction for all batch")
                    continue
                
                if i==0:
                    cln_log_attn_1     = cln_log_attn_1[correctly_predicted]
                    cln_log_attn_2     = cln_log_attn_2[correctly_predicted]

                correct_batch_size = correctly_predicted.sum()
                               
                # Adversarial Perturbation
                batch_perturbation      = self.perturbation_per_class[batch_label]
               
                batch_adv_image = self.add_perturbation(batch_image.reshape(-1,3,224,224),\
                                                        batch_perturbation.reshape(-1,3,224,224),\
                                                        normalized_perturbation=normalized_perturbation,\
                                                        project_perturbation=project_perturbation,
                                                        scale=True,\
                                                        norm_p=norm_p,\
                                                        norm_perturbation=norm_perturbation)

                # Attack
                adv_pred = model(batch_adv_image)
                adv_pred = torch.argmax(adv_pred, 1)
                adv_attr = model.dazle.package_out['A_p'] * model.dazle.package_out['S_p']

                num_attacked_samples    += correct_batch_size
                model_accuracy          += (adv_pred == batch_label).float().sum().detach().clone()
                model_initial_accuracy  += (pred     == batch_label).float().sum().detach().clone()

                list_of_true_classes.append(batch_label.cpu().detach().clone())
                list_of_adv_classes.append(adv_pred.cpu().detach().clone())


                # To log less images
                if i==0:
                    adv_title  = [self.class_names.name.tolist()[i] for i in(adv_pred)]
                    cln_title  = [self.class_names.name.tolist()[i] for i in(batch_label)]
                    self.image_visualization(inverse_transform(batch_image), inverse_transform(batch_perturbation),\
                                             inverse_transform(batch_adv_image), adv_title, cln_title, attack_scenario, norm_perturbation, norm_p,validation)
                    self.visualize_attention(images=batch_image, labels=pred, attentions_map=cln_log_attn_1, attentions_attribute=cln_log_attn_2,\
                                             n_top_attr=5, mode="clean")
                    
                    self.visualize_attention(images=batch_adv_image, labels=adv_pred, attentions_map=model.dazle.package_out['A'], attentions_attribute=model.dazle.package_out['A_p'],\
                                             n_top_attr=5, mode="adv", norm=norm_p)




                diff_img = inverse_transform(batch_image) - inverse_transform(batch_adv_image)        
                diff_img = diff_img.abs()
                list_diff_img.append(diff_img) 


            tensor_diff_img = torch.cat(list_diff_img, dim=0)
            avg_per_pixel_perturbation = tensor_diff_img.sum()/tensor_diff_img.size(0)/(224*224)
            validation_text = "validation" if validation else "test"
            if norm_p==2:
                self.logger.add_scalar('avg per pixel_L2/'+attack_scenario+validation_text, avg_per_pixel_perturbation,norm_perturbation)
                self.logger.add_scalar('true model accuracy_L2/'+attack_scenario+validation_text, model_accuracy/num_attacked_samples, norm_perturbation)
                self.logger.add_scalar('true model initial accuracy_L2/'+attack_scenario+validation_text, model_initial_accuracy/num_attacked_samples, norm_perturbation)
            else:
                self.logger.add_scalar('avg per pixel_Linf/'+attack_scenario+'_norm*100_'+validation_text, avg_per_pixel_perturbation,norm_perturbation*100)
                self.logger.add_scalar('true model accuracy_Linf/'+attack_scenario+'_norm*100_'+validation_text, model_accuracy/num_attacked_samples, norm_perturbation*100)
                self.logger.add_scalar('true model initial accuracy_Linf/'+attack_scenario+'_norm*100_'+validation_text, model_initial_accuracy/num_attacked_samples, norm_perturbation*100)




            true_classes = torch.cat(list_of_true_classes, dim=0)
            adv_classes  = torch.cat(list_of_adv_classes, dim=0)
            print("\n\n\ntrue classes size\n",true_classes.size())

            pd.DataFrame(true_classes.cpu().detach().clone().numpy()).to_csv(self.dataset+"_"+attack_scenario+'_trueClasses.csv')            
            pd.DataFrame(adv_classes.cpu().detach().clone().numpy()).to_csv(self.dataset+"_"+attack_scenario+'_advClasses.csv')            
            
            return model_accuracy/num_attacked_samples, model_initial_accuracy/num_attacked_samples
               
        
    ####################################################################
    ###              Reports the Quality                             ###
    ####################################################################

    def plot_similarity_example(self,index_1,index_2,mode, similarity, dataloader, step, plot_class=True):

        x        = self.class_names.name.tolist()
        y        = self.attr_names.name.tolist()
        fig, axs = plt.subplots(1,2)

        for j,i in enumerate([index_1,index_2]):
            if not plot_class:
                attr = i
                i    = dataloader.att[:,i].argmax()

            if i in(dataloader.seenclasses):
                idx = (torch.nonzero(dataloader.data['test_seen']['labels'].cuda()==i)[0])
                image = dataloader.data['test_seen']['images'][idx].squeeze()
                axs[j].imshow(inverse_transform(image).cpu().permute(1,2,0))
                if not plot_class:
                    axs[j].set_title("seen/"+y[attr])
                else:
                    axs[j].set_title("seen/"+x[i])

            else:
                idx = (torch.nonzero(dataloader.data['test_unseen']['labels'].cuda()==i)[0])
                image = dataloader.data['test_unseen']['images'][idx].squeeze()
                axs[j].imshow(inverse_transform(image).cpu().permute(1,2,0))
                if not plot_class:
                    axs[j].set_title("unseen/"+y[attr])
                else:
                    axs[j].set_title("unseen/"+x[i])

            fig.suptitle("similarity = "+str(similarity.item()))
        
        if not plot_class:
            self.logger.add_figure("Attribute similarity/"+mode, fig, step)
        else:
            self.logger.add_figure("Class similarity/"+mode, fig, step)




    def correlation_of_attribute_perturbations(self, dataloader):
        similarity      = torch.nn.CosineSimilarity(dim=1,eps=1e-6)
        universals      = self.e_a.reshape(self.e_a.size(0),-1)
        cosine_sim_list = []

        for i in range(universals.size(0)):
            cosine_sim_one_attr = similarity(universals[i].unsqueeze(0),universals)
            cosine_sim_list.append(cosine_sim_one_attr)
        
        self.cosine_sim_matrix   = torch.stack(cosine_sim_list).squeeze()
    
        grid_tensor = self.cosine_sim_matrix.cpu().detach().clone()
        grid    = grid_tensor.numpy()
        x       = self.attr_names.name.tolist()       
        ax      = sns.heatmap(grid,vmin=-1, vmax=1)

        plt.xticks(range(0,len(x),1), x, fontsize=0.5,rotation=90)
        plt.yticks(range(0,len(x),1), x, fontsize=0.5)
        ax.set_aspect('equal')
        fig=ax.get_figure()

        self.logger.add_figure("Correlation of attribute universals on each other", fig)
        pd.DataFrame(self.cosine_sim_matrix.cpu().detach().clone().numpy()).to_csv(self.dataset+'_correlation_matrix.csv')

        # TO plot only for wings
        self.cosine_sim_matrix_wings=self.cosine_sim_matrix[9:24,9:24]
        # print("wings matrix\n\n",self.cosine_sim_matrix_wings)
        grid_tensor = self.cosine_sim_matrix_wings.cpu().detach().clone()
        grid    = grid_tensor.numpy()
        x       = self.attr_names.name[9:24].tolist()       
        ax1      = sns.heatmap(grid,vmin=-1, vmax=1)

        plt.xticks(range(0,len(x),1), x, fontsize=0.5,rotation=90)
        plt.yticks(range(0,len(x),1), x, fontsize=0.5)
        ax1.set_aspect('equal')
        fig1=ax1.get_figure()

        self.logger.add_figure("Correlation of Wings attribute universals on each other", fig1)


        # log the largest similarity
        val, idx = torch.topk(torch.triu(grid_tensor,1).flatten(),k=20,largest=True)
        idx = (np.array(np.unravel_index(idx.numpy(), grid.shape)).T)
        idx = list(set((a,b) if a<=b else (b,a) for a,b in idx))
        for i in range(len(idx)):
            self.plot_similarity_example(index_1=idx[i][0], index_2=idx[i][1], mode="largest", similarity=grid_tensor[idx[i][0],idx[i][1]], dataloader=dataloader, step=i, plot_class=False)

        # log the smallest similarity
        val, idx = torch.topk(grid_tensor.flatten(),k=20,largest=False)
        idx = (np.array(np.unravel_index(idx.numpy(), grid.shape)).T)
        idx = list(set((a,b) if a<=b else (b,a) for a,b in idx))
        for i in range(len(idx)):
            self.plot_similarity_example(index_1=idx[i][0], index_2=idx[i][1], mode="smallest", similarity=grid_tensor[idx[i][0],idx[i][1]], dataloader=dataloader, step=i, plot_class=False)

        # log the near 0 similarity
        val, idx = torch.topk(grid_tensor.abs().flatten(),k=20,largest=False)
        idx = (np.array(np.unravel_index(idx.numpy(), grid.shape)).T)
        idx = list(set((a,b) if a<=b else (b,a) for a,b in idx))
        for i in range(len(idx)):
            self.plot_similarity_example(index_1=idx[i][0], index_2=idx[i][1], mode="near zero", similarity=grid_tensor[idx[i][0],idx[i][1]], dataloader=dataloader, step=i, plot_class=False)




    def correlation_of_class_perturbations(self, w2v_vector, semantic_vector, dataloader):
        w_comb                  = self.compute_weights(w2v_vector, semantic_vector)
        if self.num_clusters>0:
            self.perturbation_per_class  = torch.einsum('ay,achw->ychw', w_comb, self.e_a[self.clusters.attr_cluster]) 
        else:
            self.perturbation_per_class  = torch.einsum('ay,achw->ychw', w_comb, self.e_a)    

        similarity      = torch.nn.CosineSimilarity(dim=1,eps=1e-6)
        universals      = self.perturbation_per_class.reshape(self.perturbation_per_class.size(0),-1)
        cosine_sim_list = []

        for i in range(universals.size(0)):
            cosine_sim_one_attr = similarity(universals[i].unsqueeze(0),universals)
            cosine_sim_list.append(cosine_sim_one_attr)
        
        self.cosine_sim_matrix   = torch.stack(cosine_sim_list).squeeze()
    
        grid_tensor = self.cosine_sim_matrix.cpu().detach().clone()
        grid    = grid_tensor.numpy()
        x       = self.class_names.name.tolist()       
        ax      = sns.heatmap(grid,vmin=-1, vmax=1)

        plt.xticks(range(0,len(x),1), x, fontsize=0.5,rotation=90)
        plt.yticks(range(0,len(x),1), x, fontsize=0.5)
        ax.set_aspect('equal')
        fig = ax.get_figure()

        self.logger.add_figure("Correlation of class universals on each other", fig)

        # log the largest similarity
        val, idx = torch.topk(torch.triu(grid_tensor,1).flatten(),k=20,largest=True)
        idx = (np.array(np.unravel_index(idx.numpy(), grid.shape)).T)
        idx = list(set((a,b) if a<=b else (b,a) for a,b in idx))
        for i in range(len(idx)):
            self.plot_similarity_example(index_1=idx[i][0], index_2=idx[i][1], mode="largest", similarity=grid_tensor[idx[i][0],idx[i][1]], dataloader=dataloader, step=i)
            # self.logger.add_text("Similarity of classes/largest", x[idx[i][0]]+" , "+x[idx[i][1]]+" = "+str((grid_tensor[idx[i][0],idx[i][1]]).item()), i)

        # log the smallest similarity
        val, idx = torch.topk(grid_tensor.flatten(),k=20,largest=False)
        idx = (np.array(np.unravel_index(idx.numpy(), grid.shape)).T)
        idx = list(set((a,b) if a<=b else (b,a) for a,b in idx))
        for i in range(len(idx)):
            self.plot_similarity_example(index_1=idx[i][0], index_2=idx[i][1], mode="smallest", similarity=grid_tensor[idx[i][0],idx[i][1]], dataloader=dataloader, step=i)
            # self.logger.add_text("Similarity of classes/smallest", x[idx[i][0]]+" , "+x[idx[i][1]]+" = "+str((grid_tensor[idx[i][0],idx[i][1]]).item()), i)

        # log the near 0 similarity
        val, idx = torch.topk(grid_tensor.abs().flatten(),k=20,largest=False)
        idx = (np.array(np.unravel_index(idx.numpy(), grid.shape)).T)
        idx = list(set((a,b) if a<=b else (b,a) for a,b in idx))
        for i in range(len(idx)):
            self.plot_similarity_example(index_1=idx[i][0], index_2=idx[i][1], mode="near zero", similarity=grid_tensor[idx[i][0],idx[i][1]], dataloader=dataloader, step=i)
            # self.logger.add_text("Similarity of classes/near Zero", x[idx[i][0]]+" , "+x[idx[i][1]]+" = "+str((grid_tensor[idx[i][0],idx[i][1]]).item()), i)




    def major_attribute_universals(self,k=20):
        if self.num_clusters>0:
            k = min(k,self.num_clusters)
        norms           = torch.norm(self.e_a.reshape(self.e_a.size(0),-1), p=2, dim=-1)
        value, index    = torch.topk(norms,k=k, largest=True)
        index           = index.cpu()
        titles          = self.attr_names.iloc[index]
        images          = self.e_a[index]
        if k==20:
            fig, axes       = plt.subplots(4,5,figsize=(30,30))  
        else:
            fig, axes       = plt.subplots((k),figsize=(30,30))
        for i, ax in enumerate(axes.ravel()):
            ax.set_title(titles.name.iloc[i]) 
            ax.imshow(torch.clamp(10*images[i].squeeze().cpu().detach().clone().permute(1,2,0), min=0, max=1))
        self.logger.add_figure("Major attribute universals with highest L2  x10", fig)




    def major_class_universals(self, w2v_vector, semantic_vector, k=20):
        w_comb                  = self.compute_weights(w2v_vector, semantic_vector)
        if self.num_clusters>0:
            self.perturbation_per_class  = torch.einsum('ay,achw->ychw', w_comb, self.e_a[self.clusters.attr_cluster]) 
        else:
            self.perturbation_per_class  = torch.einsum('ay,achw->ychw', w_comb, self.e_a)

        norms           = torch.norm(self.perturbation_per_class.reshape(self.perturbation_per_class.size(0),-1), p=2, dim=-1)
        value, index    = torch.topk(norms,k=k, largest=True)
        index           = index.cpu()
        titles          = self.class_names.iloc[index]
        images          = self.perturbation_per_class[index]
        if k==20:
            fig, axes       = plt.subplots(4,5,figsize=(30,30))  
        else:
            fig, axes       = plt.subplots((k),figsize=(30,30))
        for i, ax in enumerate(axes.ravel()):
            ax.set_title(titles.name.iloc[i]) 
            ax.imshow(torch.clamp(10*images[i].squeeze().cpu().detach().clone().permute(1,2,0), min=0, max=1))
        self.logger.add_figure("Major class combined-universals with highest L2  x10", fig)




    def major_weights_of_universals(self, w2v_vector, semantic_vector, k=20):
        w_comb          = self.compute_weights(w2v_vector, semantic_vector)
        value, index    = torch.topk(w_comb.reshape(-1).abs(),k=k, largest=True)
        index           = index.cpu()
        w_prime         = torch.zeros_like(w_comb.reshape(-1))
        w_prime[index]  = w_comb.reshape(-1)[index]

        ax      = sns.heatmap(w_comb.cpu().detach().clone().numpy(),vmin=-1, vmax=1)       
        fig     = ax.get_figure()
        self.logger.add_figure("Weights & Major weights", fig,0)

        ax      = sns.heatmap(w_prime.reshape(w_comb.size(0),-1).cpu().detach().clone().numpy(),vmin=-1, vmax=1)       
        fig     = ax.get_figure()
        self.logger.add_figure("Weights & Major weights", fig,1)



    
    def graphs_class_attribute_relation(self, dataloader, w2v_vector, semantic_vector, k=5):
        w_comb          = self.compute_weights(w2v_vector, semantic_vector)
        for i in range(10):#(w_comb.size(1)):
            G       = nx.Graph()
            curr_w  = w_comb[:,i].abs().detach().clone()
            value, index    = torch.topk(curr_w,k=k, largest=True)
            index           = index.cpu()
            value           = value.cpu()
            curr_class      = (self.class_names.iloc[i]).values[0]
            curr_attr       = (self.attr_names.iloc[index,0]).values.tolist()

            G.add_node(curr_class)
            G.add_nodes_from(curr_attr)
            G.add_weighted_edges_from([(curr_class, curr_attr[j], value[j]) for j in range(len(index))])

            edge_width = [100*G[u][v]['weight'] for u,v in G.edges()]
            edge_color = [int(100*G[u][v]['weight']) for u,v in G.edges()]
            
            # fig = plt.figure(figsize=(10,10))
            # nx.draw_networkx(G,\
            #                     node_color ='purple',\
            #                     alpha=0.7,\
            #                     with_labels=True, width=edge_width, edge_color=edge_color, cmap=plt.cm.Blues)
            # plt.axis('off')
            # plt.tight_layout()
            # self.logger.add_figure("Graphs class-attribute relations", fig,i)

            fig = plt.figure(figsize=(90,15))
            fig.add_subplot(1,2,1)
            nx.draw_networkx(G,\
                                node_color ='purple',\
                                alpha=0.7,\
                                with_labels=True, width=edge_width, edge_color=edge_color, cmap=plt.cm.Blues, font_size=20)
            
            # plot image
            fig.add_subplot(1,2,2)
            
            if i in(dataloader.seenclasses):
                idx = (torch.nonzero(dataloader.data['test_seen']['labels'].cuda()==i)[0])
                image = dataloader.data['test_seen']['images'][idx].squeeze()
                plt.imshow(inverse_transform(image).cpu().permute(1,2,0))
                plt.title("seen")

            else:
                idx = (torch.nonzero(dataloader.data['test_unseen']['labels'].cuda()==i)[0])
                image = dataloader.data['test_unseen']['images'][idx].squeeze()
                plt.imshow(inverse_transform(image).cpu().permute(1,2,0))
                plt.title("unseen")

            plt.axis('off')
            plt.tight_layout()
            self.logger.add_figure("True Graphs class-attribute relations", fig,i)



    def utility_over_w_a_c(self,w2v_vector, semantic_vector):
        fig, ax = plt.subplots(1,2,figsize=(50,20))
        for m in range(2):
            
            w_comb       = self.compute_weights(w2v_vector, semantic_vector)
            val,idx      = torch.sort(w_comb.abs().sum(dim=m),descending=True)
            val_selected = torch.cat([val[0:20], val[-20:]]).cpu().detach().numpy()
            idx_selected = torch.cat([idx[0:20], idx[-20:]]).cpu().detach().numpy()

            if m==0:
                ax[m].barh([self.class_names.name.tolist()[i] for i in(idx_selected)],val_selected)
                ax[m].set_title("utility of classes")
            else:
                ax[m].barh([self.attr_names.name.tolist()[i] for i in(idx_selected)],val_selected)
                ax[m].set_title("utility of attributes")

        self.logger.add_figure("Utilities",fig)


    def visualize_attention(self, images, labels, attentions_map, attentions_attribute, n_top_attr, mode="clean", norm=0):
        # print("attn map sizes ",attentions_map.size(),attentions_map[0].size())
        h = int(attentions_map.size(-1)**0.5)
        for i in range(min(10,images.size(0))):#(images.size(0)):
            image   = images[i]
            label   = labels[i]
            attn_1  = attentions_map[i]
            attn_2  = attentions_attribute[i]
            image   = inverse_transform(image).detach().clone().cpu().permute(1,2,0)
            fig, ax = plt.subplots(1, n_top_attr+1, figsize=(30,30))
            ax[0].set_title(self.class_names.name.tolist()[label])
            ax[0].imshow(image)

            idx_top = torch.argsort(-attn_2)[:n_top_attr]
            # print("attributes, values",idx_top, attn_2)
            
            for idx_ctxt, idx_attr in enumerate(idx_top):
                ax[idx_ctxt+1].imshow(image)
                attn_curr   = attn_1[idx_attr,:].reshape(h,h)
                attn_image  = skimage.transform.pyramid_expand(attn_curr.cpu().detach().clone(), upscale= image.size(0)/h,\
                                                               sigma=10, multichannel=False)
                ax[idx_ctxt+1].imshow(attn_image, alpha=0.7)
                if mode=="clean":
                    ax[idx_ctxt+1].set_title(str(self.attr_names.name.tolist()[idx_attr])+ "/ "+ str(round(attn_2[idx_attr].item(),2)))
                else:
                    ax[idx_ctxt+1].set_title(str(self.attr_names.name.tolist()[idx_attr])+ "/ "+ str(round(attn_2[idx_attr].item(),2))+"/ norm"+str(norm.item()))
            
            fig.tight_layout()

            if mode=="clean":
                self.logger.add_figure("Attentions over Attributes/"+mode,fig,i)
            else:
                self.logger.add_figure("Attentions over Attributes/"+mode+"_norm"+str(norm.item()),fig,i)
            
            # time.sleep(3)

            fig = plt.figure(figsize=(10,10))
            plt.bar(torch.arange(attn_2.size(-1)),attn_2.cpu())
            self.logger.add_figure("Attentions over Attributes/attr_attrn_"+mode+"_norm"+str(norm),fig)













    
        
