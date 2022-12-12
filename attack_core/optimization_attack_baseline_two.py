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
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch.nn as nn
import scipy
from global_setting import NFS_path_AoA,save_NFS
import sys
import random
sys.path.insert(1,'/mnt/raptor/nasim/early_stopping/early-stopping-pytorch')
from pytorchtools import EarlyStopping

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


# Universal Generation and Attack
class universal_end2end_attack(torch.nn.Module):
    def __init__(self, num_attributes, num_classes, image_size, w2v_size, dataset, w_as="function",\
                                        device=torch.device("cpu"), learning_rate=0.001, log_folder="./log_baselines/", log_title="_", if_dropout=False):

        super(universal_end2end_attack, self).__init__()
        self.num_attributes = num_attributes
        self.num_classes    = num_classes
        self.w_as = w_as

        self.w_a_c      = torch.ones(num_attributes,num_classes, device= device, requires_grad=False)/num_attributes
       
        self.e_a        = torch.nn.Parameter(torch.zeros(num_attributes,\
                                            image_size[0], image_size[1],\
                                            image_size[2], device=device))
       
        self.optimizer  = torch.optim.Adam([self.e_a], lr=learning_rate)
       
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

       
        
        torch.nn.init.uniform_(self.e_a, a=0.00001, b=0.001)
   
   
   
    def attribute_score(self, model, batch_image):    
        pred        = model(batch_image)
        target_label= torch.min(pred, 1)[1] # for CE loss min
        # target_label= torch.topk(pred, 2,dim=-1,largest=True)[1][:,1] # for CE loss 2nd max
        pred        = torch.argmax(pred, 1)
        attr_score  = model.dazle.package_out['A_p'] * model.dazle.package_out['S_p']
        return attr_score, pred, target_label
   
   
   
    def compute_weights(self, w2v_vector, semantic_vector):
        if self.w_as=="function":
            # Calculate Weights of perturbations combination
            w_comb              = torch.einsum('aw,aw,ya->ay', w2v_vector, self.w_a_c, semantic_vector)
            w_comb_normalized   = torch.tanh(w_comb)
            #inv_norm    = torch.norm(w_comb, p=2, dim=0).pow(-1)
            #w_comb_normalized = torch.einsum("y,ay->ay", inv_norm, w_comb)
            #w_comb  = F.softmax(w_comb+0.1, dim=0)
           
        elif self.w_as=="variable":
            w_comb_normalized   = self.w_a_c.detach().clone()
            #w_comb_normalized   = torch.tanh(w_comb)
            # inv_norm    = torch.norm(w_comb, p=2, dim=0).pow(-1)
            # w_comb_normalized = torch.einsum("y,ay->ay", inv_norm, w_comb)
        
        elif self.w_as=="uniform":
            w_comb_normalized = self.w_a_c.detach().clone()

           
        return w_comb_normalized
   
   
   
    def log_function(self, iter, loss, loss_1, loss_2):
        self.logger.add_scalar('Loss/train', loss, iter)
        self.logger.add_scalar('Loss1/train', loss_1, iter)
        self.logger.add_scalar('Loss2/train', loss_2, iter)
       
        if iter%50==0:
            img_grid = torchvision.utils.make_grid(self.e_a)
            self.logger.add_image('universals per attribute', img_grid, iter)

   
   
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
                normalized_perturbation=True, project_perturbation=False, checkpoint_path="./"):
        # Initialize Variables
        dataset_length      = len(dataloader['images'])
        dataset_length_val  = len(dataloader_val['images'])
        iter=0

        early_stopping = EarlyStopping(patience=10, verbose=True, path=checkpoint_path)#20
       
        # Iterate over Epochs
        for epoch in range(num_epochs):

            loss_train  =[]
            loss_train_1=[]
            loss_train_2=[]
            #Iterate over batch
            for i in range(0, dataset_length, batch_size):
               
                self.optimizer.zero_grad()
               
                batch_label   = dataloader['labels'][i:i+batch_size].to(self.device)
                batch_image   = dataloader['images'][i:i+batch_size].to(self.device)

                # Calculate weights for combining universals
                w_comb = self.compute_weights(w2v_vector, semantic_vector)
               
                # Calculate Adversarial Perturbations Per Class
                e = torch.einsum('ay,achw->ychw', w_comb, self.e_a)

                # Calculate attribute scores
                attribute_score, c, target_label = self.attribute_score(model, batch_image)
               
                # Filter only the correct samples in the batch
                corr_batch_image = batch_image[torch.nonzero(c==batch_label)]
                corr_batch_label = batch_label[torch.nonzero(c==batch_label)]
               
                # No Correct Prediction To Attack
                if (c==batch_label).sum()<=0:
                    continue

                # Create Perturbations for the batch
                adv_batch_image = self.add_perturbation(corr_batch_image.reshape(-1,3,224,224),\
                                                        e[corr_batch_label].reshape(-1,3,224,224),\
                                                        normalized_perturbation=normalized_perturbation,\
                                                        project_perturbation=project_perturbation, scale=False)
                # Attack
                adv_attribute_score, adv_c, _  = self.attribute_score(model, adv_batch_image.reshape(-1,3,224,224))
                final_adv_scores            = torch.einsum('ba,ya->by', adv_attribute_score, semantic_vector)
               
                # Calculate Loss function
                if loss_norm==2:
                    loss_1  = 0.5 * torch.norm(self.e_a.reshape(self.e_a.size(0),-1), p=loss_norm, dim=-1).pow(loss_norm).sum()#/ (self.num_attributes)
                else:
                    # loss_1  = 0.5 * torch.norm(self.e.reshape(self.e.size(0),-1), p=loss_norm, dim=-1).sum()
                    t = (0.005) / 0.224
                    loss_1  = torch.einsum('achw,achw->achw', (self.e_a.abs() - t )>0 , self.e_a.abs() - t ).sum()                
                
               
                loss_2  = self.loss_adversraial_condition(final_adv_scores, corr_batch_label, confidence)

                #cross entropy loss
                # loss_2 = torch.log(self.criterion(final_adv_scores, target_label[torch.nonzero(c==batch_label)].reshape(-1)))
                #regularization = torch.norm(e.reshape(e.size(0),-1), p=1, dim=-1).sum()
                loss    = norm_coefficient*loss_1 + loss_coefficient * loss_2 #+0.0001*regularization
                loss_train_1.append(loss_1.item())
                loss_train_2.append(loss_2.item())
                loss_train.append(loss.item())
                # Backpropagate
                loss.backward()
                clipping_value = 1 
                torch.nn.utils.clip_grad_norm([self.e_a], clipping_value)

                self.optimizer.step()               
                iter+=1
            
            # Log values
            self.log_function(epoch, sum(loss_train)/len(loss_train), sum(loss_train_1)/len(loss_train_1), sum(loss_train_2)/len(loss_train_2))

            ## Validation step
            #Iterate over batch
            with torch.no_grad():
                loss_val =[]
                loss_val_1=[]
                loss_val_2=[]
                for i in range(0, dataset_length_val, batch_size):
                
                    batch_label   = dataloader_val['labels'][i:i+batch_size].to(self.device)
                    batch_image   = dataloader_val['images'][i:i+batch_size].to(self.device)
              
                    # Calculate weights for combining universals
                    w_comb = self.compute_weights(w2v_vector, semantic_vector)
               
                    # Calculate Adversarial Perturbations Per Class
                    e = torch.einsum('ay,achw->ychw', w_comb, self.e_a)

                    # Calculate attribute scores
                attribute_score, c, target_label = self.attribute_score(model, batch_image)
               
                # Filter only the correct samples in the batch
                corr_batch_image = batch_image[torch.nonzero(c==batch_label)]
                corr_batch_label = batch_label[torch.nonzero(c==batch_label)]
                
                # No correct prediction to attack
                if (c==batch_label).sum()<=0:
                    continue

                # Create Perturbations for the batch
                adv_batch_image = self.add_perturbation(corr_batch_image.reshape(-1,3,224,224),\
                                                        e[corr_batch_label].reshape(-1,3,224,224),\
                                                        normalized_perturbation=normalized_perturbation,\
                                                        project_perturbation=project_perturbation, scale=False)
                # Attack
                adv_attribute_score, adv_c, _  = self.attribute_score(model, adv_batch_image.reshape(-1,3,224,224))
                final_adv_scores            = torch.einsum('ba,ya->by', adv_attribute_score, semantic_vector)
               
                # Calculate Loss function
                if loss_norm==2:
                    loss_1  = 0.5 * torch.norm(self.e_a.reshape(self.e_a.size(0),-1), p=loss_norm, dim=-1).pow(loss_norm).sum()
                else:
                    # loss_1  = 0.5 * torch.norm(self.e.reshape(self.e.size(0),-1), p=loss_norm, dim=-1).sum()
                    t = (0.005 )/0.224
                    loss_1  = torch.einsum('achw,achw->achw', (self.e_a.abs() - t )>0 , self.e_a.abs() - t ).sum()


                loss_2  = self.loss_adversraial_condition(final_adv_scores, corr_batch_label, confidence)
                #cross entropy loss
                # loss_2 = torch.log(self.criterion(final_adv_scores, target_label[torch.nonzero(c==batch_label)].reshape(-1)))
                #regularization = torch.norm(e.reshape(e.size(0),-1), p=1, dim=-1).sum()
                loss    = norm_coefficient*loss_1 + loss_coefficient * loss_2 #+0.0001*regularization
                loss_val.append(loss.item())
                loss_val_1.append(loss_1.item())
                loss_val_2.append(loss_2.item())
                
            self.logger.add_scalar('Loss1/validation', sum(loss_val_1)/len(loss_val_1), epoch)
            self.logger.add_scalar('Loss2/validation', sum(loss_val_2)/len(loss_val_2), epoch)
            self.logger.add_scalar('Loss/validation', sum(loss_val)/len(loss_val), epoch)

            early_stopping( sum(loss_val)/len(loss_val), self)
            if early_stopping.early_stop:
                print("Early stopping at epoch\t", epoch)
                break
        # load the last checkpoint with the best model
        self.load_state_dict(torch.load(checkpoint_path))
        return    




   
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
            self.perturbation_per_class  = torch.einsum('ay,achw->ychw', w_comb, self.e_a)
            pert_grid = torchvision.utils.make_grid(self.perturbation_per_class)
            self.logger.add_image("perturbation_per_class",pert_grid)
            pert_grid = torchvision.utils.make_grid(self.e_a)
            self.logger.add_image("universals per attribute/test",pert_grid)
           
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
               
               
               # To log less images
                if i==0:
                    adv_title  = [self.class_names.name.tolist()[i] for i in(adv_pred)]
                    cln_title  = [self.class_names.name.tolist()[i] for i in(batch_label)]
                    self.image_visualization(inverse_transform(batch_image), inverse_transform(batch_perturbation),\
                                             inverse_transform(batch_adv_image), adv_title, cln_title, attack_scenario, norm_perturbation, norm_p, validation)
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
            
            return model_accuracy/num_attacked_samples, model_initial_accuracy/num_attacked_samples

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















