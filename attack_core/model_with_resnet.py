
import torchvision.models.resnet as models
import torch
import torch.nn as nn
import sys
# sys.path.insert(1,'/home/nasim/adv_fine_repo/adversarial-fine-grained/adversarial_fine_grained/fine_grained_code_/neurIPS20_CompositionZSL')
from core.DAZLE import DAZLE
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--nz', type=int, default=312, help='size of the latent z vector')
# parser.add_argument('--normalize_V', type=bool, default=False, help='normalize_V')
parser.add_argument('--gen_hidden', type=int, default=1024, help='gen_hidden')
opt = parser.parse_known_args()[0] #omit unknown arguments


class Resnet_DAZLE(nn.Module):  

    def __init__(self, dataloader, device, dataset):
        super(Resnet_DAZLE, self).__init__()
        # init resenet
        model_ref = models.resnet101(pretrained=True)
        model_ref.eval()

        self.resnet = nn.Sequential(*list(model_ref.children())[:-2])
        self.resnet.to(device)
        self.resnet.eval()
        for param in self.resnet.parameters():
            param.requires_grad = False

        
        # init dazle
        dim_f = 2048
        dim_v = 300
        init_w2v_att = dataloader.w2v_att
        att = dataloader.att#dataloader.normalize_att#
        normalize_att = dataloader.normalize_att

        trainable_w2v = True
        lambda_1 = 0.0
        lambda_2 = 0.0
        lambda_3 = 0.0
        bias = 0
        prob_prune = 0
        uniform_att_1 = False
        uniform_att_2 = False

        seenclass = dataloader.seenclasses
        unseenclass = dataloader.unseenclasses
        desired_mass = 1#unseenclass.size(0)/(seenclass.size(0)+unseenclass.size(0))

        if dataset =="AWA2":
            norm_v = True
        else:
            norm_v=False

        dazle_model = DAZLE(dim_f,dim_v,init_w2v_att,att,normalize_att,
                            seenclass,unseenclass,
                            lambda_1,lambda_2,lambda_3,
                            device,
                            trainable_w2v,normalize_V=norm_v,normalize_F=True,is_conservative=False,
                            uniform_att_1=uniform_att_1,uniform_att_2=uniform_att_2,
                            prob_prune=prob_prune,desired_mass=desired_mass, is_conv=False,
                            is_bias=True)
                            
       
        save_path="./model/nips_dazle++_"+dataset.lower()+"_nonUniformAttention_bias.pt"
        dazle_model=torch.load(save_path,map_location=device)

        self.dazle   = dazle_model
        
        
    def forward(self,img):
        feature            = self.resnet(img)
        out_after_softmax  = self.dazle(feature)
        out_before_softmax = self.dazle.package_out['S_pp']
        
        return out_before_softmax
    
    
    def forward_attribute(self, img, attr_idx):
        feature            = self.resnet(img)
        out_after_softmax  = self.dazle(feature)
        
        # attended attribute = attention beta * atteribute e_i
        attended_attribute = self.dazle.package_out['A_p'] * self.dazle.package_out['S_p']

        return attended_attribute[:,attr_idx]

    
