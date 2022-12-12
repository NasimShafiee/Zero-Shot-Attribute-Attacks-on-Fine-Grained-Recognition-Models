
from typing_extensions import final
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
import sys
from global_setting import NFS_path_AoA,save_NFS

sys.path.insert(1,'/mnt/raptor/nasim/early_stopping/early-stopping-pytorch')
from pytorchtools import EarlyStopping

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


##################################################################################
##################################################################################
##################################################################################

class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, norm_type, act_type='selu', use_dropout=False, n_blocks=6, padding_type='reflect', gpu_ids=[]):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()

        self.name = 'resnet'
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpulist = gpu_ids
        self.gpu_ids=gpu_ids
        self.num_gpus = 1#len(self.gpulist)

        use_bias = norm_type == 'instance'

        if norm_type == 'batch':
            norm_layer = nn.BatchNorm2d
        elif norm_type == 'instance':
            norm_layer = nn.InstanceNorm2d

        if act_type == 'selu':
            self.act = nn.SELU(True)
        else:
            self.act = nn.ReLU(True)

        model0 = [nn.ReflectionPad2d(3),
                  nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                            bias=use_bias),
                  norm_layer(ngf),
                  self.act]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model0 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                 stride=2, padding=1, bias=use_bias),
                       norm_layer(ngf * mult * 2),
                       self.act]

        if self.num_gpus == 1:
            mult = 2**n_downsampling
            for i in range(n_blocks):
                model0 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        elif self.num_gpus == 2:
            model1 = []
            mult = 2**n_downsampling
            mid = int(n_blocks / 2)
            for i in range(mid):
                model0 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
            for i in range(n_blocks - mid):
                model1 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        elif self.num_gpus == 3:
            model1 = []
            model2 = []
            mult = 2**n_downsampling
            mid1 = int(n_blocks / 5)
            mid2 = mid1 + int((n_blocks - mid1) / 4.0 * 3)
            # mid = int(n_blocks / 2)
            for i in range(mid1):
                model0 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
            for i in range(mid1, mid2):
                model2 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
            for i in range(mid2, n_blocks):
                model1 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        if self.num_gpus >= 2:
            for i in range(n_downsampling):
                mult = 2**(n_downsampling - i)
                model1 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                            kernel_size=3, stride=2,
                                            padding=1, output_padding=1,
                                            bias=use_bias),
                        norm_layer(int(ngf * mult / 2)),
                        self.act]
            model1 += [nn.ReflectionPad2d(3)]
            model1 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
            model1 += [nn.Tanh()]
        else:
            for i in range(n_downsampling):
                mult = 2**(n_downsampling - i)
                model0 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                            kernel_size=3, stride=2,
                                            padding=1, output_padding=1,
                                            bias=use_bias),
                        norm_layer(int(ngf * mult / 2)),
                        self.act]
            model0 += [nn.ReflectionPad2d(3)]
            model0 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
            model0 += [nn.Tanh()] 

        self.model0 = nn.Sequential(*model0)
        self.model0.to(gpu_ids)
        if self.num_gpus == 2:
            self.model1 = nn.Sequential(*model1)
            self.model1.to(gpu_ids)
        if self.num_gpus == 3:
            self.model2 = nn.Sequential(*model2)
            self.model2.to(gpu_ids)

    def forward(self, input):
        # input = input.to(self.gpu_ids)
        input = self.model0(input)
        if self.num_gpus == 3:
            # input = input.to(self.gpu_ids)
            input = self.model2(input)
        if self.num_gpus == 2:
            # input = input.to(self.gpu_ids)
            input = self.model1(input)
        return input


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

def weights_init(m, act_type='relu'):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if act_type == 'selu':
            n = float(m.in_channels * m.kernel_size[0] * m.kernel_size[1])
            m.weight.data.normal_(0.0, 1.0 / math.sqrt(n))
        else:
            m.weight.data.normal_(0.0, 0.02)        
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


##################################################################################
##################################################################################
##################################################################################


class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, norm_type='batch', act_type='selu'):
        super(UnetGenerator, self).__init__()
        self.name = 'unet'
        self.conv1 = nn.Conv2d(input_nc, ngf, 4, 2, 1)
        self.conv2 = nn.Conv2d(ngf, ngf * 2, 4, 2, 1)
        self.conv3 = nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1)
        self.conv4 = nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1)
        self.conv5 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.conv6 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.conv7 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.conv8 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.dconv1 = nn.ConvTranspose2d(ngf * 8, ngf * 8, 5, 2, 1)
        self.dconv2 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1)
        self.dconv3 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1)
        self.dconv4 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1)
        self.dconv5 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, 4, 2, 1)
        self.dconv6 = nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, 4, 2, 1)
        self.dconv7 = nn.ConvTranspose2d(ngf * 2 * 2, ngf, 4, 2, 1)
        self.dconv8 = nn.ConvTranspose2d(ngf * 2, output_nc, 4, 2, 1)

        if norm_type == 'batch':
            self.norm = nn.BatchNorm2d(ngf)
            self.norm2 = nn.BatchNorm2d(ngf * 2)
            self.norm4 = nn.BatchNorm2d(ngf * 4)
            self.norm8 = nn.BatchNorm2d(ngf * 8)
        elif norm_type == 'instance':
            self.norm = nn.InstanceNorm2d(ngf)
            self.norm2 = nn.InstanceNorm2d(ngf * 2)
            self.norm4 = nn.InstanceNorm2d(ngf * 4)
            self.norm8 = nn.InstanceNorm2d(ngf * 8)
        self.leaky_relu = nn.LeakyReLU(0.2, True)

        if act_type == 'selu':
            self.act = nn.SELU(True)
        else:
            self.act = nn.ReLU(True)

        self.dropout = nn.Dropout(0.5)

        self.tanh = nn.Tanh()

    def forward(self, input):
        # Encoder
        # Convolution layers:
        # input is (nc) x 512 x 1024
        e1 = self.conv1(input)
        # state size is (ngf) x 256 x 512
        e2 = self.norm2(self.conv2(self.leaky_relu(e1)))
        # state size is (ngf x 2) x 128 x 256
        e3 = self.norm4(self.conv3(self.leaky_relu(e2)))
        # state size is (ngf x 4) x 64 x 128
        e4 = self.norm8(self.conv4(self.leaky_relu(e3)))
        # state size is (ngf x 8) x 32 x 64
        e5 = self.norm8(self.conv5(self.leaky_relu(e4)))
        # state size is (ngf x 8) x 16 x 32
        e6 = self.norm8(self.conv6(self.leaky_relu(e5)))
        # state size is (ngf x 8) x 8 x 16
        # print("e5,e6 ",e5.size(), e6.size())
        # e7 = self.norm8(self.conv7(self.leaky_relu(e6)))
        # state size is (ngf x 8) x 4 x 8
        # No batch norm on output of Encoder
        # print("\n\ne7 size ",e7.size())
        # e8 = self.conv8(self.leaky_relu(e7))

        # Decoder
        # Deconvolution layers:
        # state size is (ngf x 8) x 2 x 4
        # d1_ = self.dropout(self.norm8(self.dconv1(self.act(e8))))
        # state size is (ngf x 8) x 4 x 8
        # d1 = torch.cat((d1_, e7), 1)
        # d2_ = self.dropout(self.norm8(self.dconv2(self.act(d1))))
        # state size is (ngf x 8) x 8 x 16
        # d2 = torch.cat((d2_, e6), 1)
        # d3_ = self.dropout(self.norm8(self.dconv3(self.act(d2))))
        # state size is (ngf x 8) x 16 x 32
        d3__=self.dropout(self.norm8(self.dconv1(self.act(e6))))
        # print("d3,e5,e6 ",d3__.size(),e5.size(),e6.size())
        d3 = torch.cat((d3__, e5), 1)
        d4_ = self.norm8(self.dconv4(self.act(d3)))
        # state size is (ngf x 8) x 32 x 64
        d4 = torch.cat((d4_, e4), 1)
        d5_ = self.norm4(self.dconv5(self.act(d4)))
        # state size is (ngf x 4) x 64 x 128
        d5 = torch.cat((d5_, e3), 1)
        d6_ = self.norm2(self.dconv6(self.act(d5)))
        # state size is (ngf x 2) x 128 x 256
        d6 = torch.cat((d6_, e2), 1)
        d7_ = self.norm(self.dconv7(self.act(d6)))
        # state size is (ngf) x 256 x 512
        d7 = torch.cat((d7_, e1), 1)
        d8 = self.dconv8(self.act(d7))
        # state size is (nc) x 512 x 1024
        output = self.tanh(d8)
        return output



# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class RecursiveUnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf, norm_type,
                 act_type='selu', use_dropout=False, gpu_ids=[]):
        super(RecursiveUnetGenerator, self).__init__()
        self.name = 'unet-rec'
        self.gpu_ids = gpu_ids

        if norm_type == 'batch':
            norm_layer = nn.BatchNorm2d
        elif norm_type == 'instance':
            norm_layer = nn.InstanceNorm2d

        if act_type == 'selu':
            self.act = nn.SELU(True)
        else:
            self.act = nn.ReLU(True)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, self.act, self.gpu_ids, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, self.act, self.gpu_ids, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, self.act, self.gpu_ids, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, self.act, self.gpu_ids, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, self.act, self.gpu_ids, input_nc=None, submodule=unet_block, norm_layer=norm_layer)

        unet_block = UnetSkipConnectionBlock(output_nc, ngf, self.act, self.gpu_ids, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        # if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
        #     return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        # else:
        return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, act, gpu_ids, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.gpulist = gpu_ids
        use_bias = norm_layer == 'instance'
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = act
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]

            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        if self.outermost:
            self.model0 = nn.Sequential(*down)
            self.model0.cuda(self.gpulist[0])
            self.model1 = submodule
            self.model1.cuda(self.gpulist[1])
            self.model2 = nn.Sequential(*up)
            self.model2.cuda(self.gpulist[0])
        else:
            self.model = nn.Sequential(*model)
            self.model.cuda(self.gpulist[1])

    def forward(self, x):
        if self.outermost:
            x = x.cuda(self.gpulist[0])
            x0 = self.model0(x).cuda(self.gpulist[1])
            x1 = self.model1(x0).cuda(self.gpulist[0])
            x2 = self.model2(x1)
            return x2
        else:
            return torch.cat([x, self.model(x)], 1)
        


##################################################################################
##################################################################################
##################################################################################


# Universal Generation and Attack
class universal_end2end_attack(torch.nn.Module):
    def __init__(self, num_attributes, num_classes, image_size, w2v_size, dataset, w_as="function",\
                                        device=torch.device("cpu"), learning_rate=0.001, log_folder="./log_baselines/", log_title="_", if_dropout=False):

        super(universal_end2end_attack, self).__init__()

        self.net_g = ResnetGenerator(3, 3, 64, norm_type='batch', act_type='relu', gpu_ids=device)#ngf64
        # self.net_g = UnetGenerator(3, 3, 64, norm_type='batch', act_type='selu').to(device)#, gpu_ids=device)#ngf64
        self.net_g.apply(weights_init)
        self.e        = torch.zeros(1,image_size[0], image_size[1],image_size[2], device=device)
        self.num_classes    = num_classes
        self.optimizer      = torch.optim.Adam(self.net_g.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.criterion_pre  = nn.CrossEntropyLoss()
        self.criterion_pre  = self.criterion_pre.to(device)
        self.center_crop    = 224
        self.noise_data     = np.random.uniform(0, 255, self.center_crop * self.center_crop * 3)
        self.im_noise       = np.reshape(self.noise_data, (3, self.center_crop, self.center_crop))
        self.im_noise       = self.im_noise[np.newaxis, :, :, :]
        self.im_noise_tr    = np.tile(self.im_noise, (30, 1, 1, 1))
        self.noise_tr       = torch.from_numpy(self.im_noise_tr).type(torch.FloatTensor).to(device)

        self.im_noise_te = np.tile(self.im_noise, (16, 1, 1, 1))
        self.noise_te = torch.from_numpy(self.im_noise_te).type(torch.FloatTensor).to(device)
        # self.optimizer  = torch.optim.Adam([self.e], lr=learning_rate)

        
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

        
        # torch.nn.init.uniform_(self.e, a=0.00001, b=0.001)
    
    
    
    def attribute_score(self, model, batch_image):    
        pred        = model(batch_image)
        pred        = torch.argmax(pred, 1)
        attr_score  = model.dazle.package_out['A_p'] * model.dazle.package_out['S_p']
        return attr_score, pred
    
    
    
    def log_function(self, iter, loss, loss_1, loss_2):
        self.logger.add_scalar('Loss/train', loss, iter)
        self.logger.add_scalar('Loss1/train', loss_1, iter)
        self.logger.add_scalar('Loss2/train', loss_2, iter)
        
        if iter%10==0:
            img_grid = torchvision.utils.make_grid(self.e)
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
            norm_p_img   = torch.norm(perturbation,p=norm_p,dim=(1,2,3)).mean()

            # Scaling Perturbations
            scale_l = torch.tensor(perturbation_scale)
            ratio_l = norm_p_img/scale_l
            perturbation_scaled = perturbation/ratio_l

        else:

            norm_p_img = torch.norm(perturbation.reshape(-1,3,224,224),p=float('inf'),dim=(2,3)).reshape(-1,3,1,1)
            scale_l = torch.tensor(perturbation_scale)
            ratio_l = norm_p_img/scale_l
            perturbation_scaled = perturbation/ratio_l

            
        
        return perturbation_scaled



    def image_visualization(self, images, perturbations, adv_images, adv_title, cln_title, attack_scenario, norm_perturbation, norm_p):
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
        self.logger.add_figure('attack/'+attack_scenario+str(norm_perturbation)+'_'+str(norm_p.item()), fig)



    ####################################################################
    ###              Train the Universals                            ###
    ####################################################################

    def forward(self, model, dataloader, dataloader_val, semantic_vector, batch_size, confidence, num_epochs=10, loss_norm=2, norm_perturbation=1, loss_coefficient=10, normalized_perturbation=True, project_perturbation=False, checkpoint_path="./"):
        # Initialize Variables
        norm_p =loss_norm
        batch_size          = 30 # as specified in GAP paper
        dataset_length      = len(dataloader['images'])
        dataset_length_val  = len(dataloader_val['images'])
        iter=0
        
        early_stopping = EarlyStopping(patience=20, verbose=True, path=checkpoint_path)

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

                
                # Calculate Unique Adversarial Perturbations 
                # e = self.e

                # Calculate attribute scores
                # attribute_score, c = self.attribute_score(model, batch_image)
                
                # least likely class in nontargeted case
                pretrained_label_float = model(batch_image)
                _, target_label = torch.min(pretrained_label_float, 1)
                c  = torch.argmax(pretrained_label_float, 1)
                
                # Filter only the correct samples in the batch
                corr_batch_image = batch_image[torch.nonzero(c==batch_label)]
                corr_batch_label = batch_label[torch.nonzero(c==batch_label)]

                # No Correct Prediction To Attack
                if (c==batch_label).sum()<=0:
                    continue

                # generate perturbation
                # print("\n\n\n\n",self.noise_tr.size(), self.net_g)
                self.e = self.net_g(self.noise_tr[0:corr_batch_image.size(0)])

                self.net_g.zero_grad()

                # Create Perturbations for the batch
                adv_batch_image = self.add_perturbation(corr_batch_image.reshape(-1,3,224,224),\
                                                        self.e,\
                                                        normalized_perturbation=True,\
                                                        project_perturbation=True, scale=True, norm_p=norm_p, norm_perturbation=norm_perturbation)#L2,7.8
                # Attack
                # adv_attribute_score, adv_c  = self.attribute_score(model, adv_batch_image.squeeze())
                # final_adv_scores            = torch.einsum('ba,ya->by', adv_attribute_score, semantic_vector)
                output_pretrained = model(adv_batch_image)
                loss = torch.log(self.criterion_pre(output_pretrained, target_label[torch.nonzero(c==batch_label)].reshape(-1)))

                loss.backward()
                self.optimizer.step()


                # Calculate Loss function
                # loss_1  = 0.5 * torch.norm(self.e.reshape(self.e.size(0),-1), p=loss_norm, dim=-1).pow(2).sum()
                # loss_2  = self.loss_adversraial_condition(final_adv_scores, corr_batch_label, confidence)
                # #regularization = torch.norm(e.reshape(e.size(0),-1), p=1, dim=-1).sum()
                # loss    = loss_1 + loss_coefficient * loss_2 #+0.0001*regularization
                # loss_train_1.append(loss_1.item())
                # loss_train_2.append(loss_2.item())
                loss_train.append(loss.item())
                # print("===> Epoch[{}]({}/{}) loss: {:.4f}".format(epoch, i, dataset_length, loss.item()))

                # Backpropagate
                # loss.backward()
                # clipping_value = 1 
                # torch.nn.utils.clip_grad_norm(model.parameters(), clipping_value)

                # self.optimizer.step()
                iter+=1

            # Log values
            self.log_function(epoch, sum(loss_train)/len(loss_train), 0, 0)

            ## Validation step
            #Iterate over batch
            with torch.no_grad():
                loss_val =[]
                loss_val_1=[]
                loss_val_2=[]
                val_batch_size =16
                for i in range(0, dataset_length_val, val_batch_size):
                
                    batch_label   = dataloader_val['labels'][i:i+val_batch_size].to(self.device)
                    batch_image   = dataloader_val['images'][i:i+val_batch_size].to(self.device)

                    # Calculate attribute scores
                    # attribute_score, c = self.attribute_score(model, batch_image)
                    pretrained_label_float = model(batch_image)
                    _, target_label = torch.min(pretrained_label_float, 1)
                    c  = torch.argmax(pretrained_label_float, 1)
                
                    # Filter only the correct samples in the batch
                    corr_batch_image = batch_image[torch.nonzero(c==batch_label)]
                    corr_batch_label = batch_label[torch.nonzero(c==batch_label)]
                
                    # No correct prediction to attack
                    if (c==batch_label).sum()<=0:
                        continue

                    # generate perturbation
                    e = self.net_g(self.noise_tr[0:corr_batch_image.reshape(-1,3,224,224).size(0)])

                    # Create Perturbations for the batch
                    adv_batch_image = self.add_perturbation(corr_batch_image.reshape(-1,3,224,224),\
                                                            e,\
                                                            normalized_perturbation=True,\
                                                            project_perturbation=True, scale=True, norm_p=norm_p, norm_perturbation=norm_perturbation)
                    # Attack
                    # adv_attribute_score, adv_c  = self.attribute_score(model, adv_batch_image.reshape(-1,3,224,224))
                    # final_adv_scores            = torch.einsum('ba,ya->by', adv_attribute_score, semantic_vector)
                    output_pretrained = model(adv_batch_image)
                    loss = torch.log(self.criterion_pre(output_pretrained, target_label[torch.nonzero(c==batch_label)].reshape(-1)))

                    # Calculate Loss function
                    # loss_1  = 0.5 * torch.norm(self.e.reshape(self.e.size(0),-1), p=loss_norm, dim=-1).pow(2).sum()
                    # loss_2  = self.loss_adversraial_condition(final_adv_scores, corr_batch_label, confidence)
                    # #regularization = torch.norm(e.reshape(e.size(0),-1), p=1, dim=-1).sum()
                    # loss    = loss_1 + loss_coefficient * loss_2 #+0.0001*regularization
                    loss_val.append(loss.item())
                    # loss_val_1.append(loss_1.item())
                    # loss_val_2.append(loss_2.item())

                # self.logger.add_scalar('Loss1/validation', sum(loss_val_1)/len(loss_val_1), epoch)
                # self.logger.add_scalar('Loss2/validation', sum(loss_val_2)/len(loss_val_2), epoch)
                self.e = self.net_g(self.noise_tr)
                self.logger.add_scalar('Loss/validation', sum(loss_val)/len(loss_val), epoch)

                # pert_grid = torchvision.utils.make_grid(self.net_g(self.noise_tr))
                # self.logger.add_image("train perturbation",pert_grid,epoch)

                early_stopping( sum(loss_val)/len(loss_val), model)
                if early_stopping.early_stop:
                    print("Early stopping at epoch\t", epoch)
                    break

        # load the last checkpoint with the best model
        model.load_state_dict(torch.load(checkpoint_path))
        return    


        
        
    def attack_with_universals(self, model, dataloader, semantic_vector, w2v_vector,\
                               batch_size, norm_perturbation=1, norm_p=1, attack_scenario="seen", normalized_perturbation=True, project_perturbation=False):
        self.net_g.eval()
        with torch.no_grad(): 
            length_dataset          = dataloader['resnet_features'].size(0)
            num_attacked_samples    = 0
            model_accuracy          = 0
            model_initial_accuracy  = 0
            list_diff_img = []
            
            # Calculate Adversarial Perturbations
            e = self.net_g(self.noise_te)
            pert_grid = torchvision.utils.make_grid(e)
            self.logger.add_image("perturbation_per_class",pert_grid)
            
            # Iterate over dataset
            batch_size=16
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
                batch_perturbation      = e[0:correct_batch_size]

                
                batch_adv_image = self.add_perturbation(batch_image.reshape(-1,3,224,224),\
                                                        batch_perturbation,\
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
                    self.image_visualization(inverse_transform(batch_image), inverse_transform(batch_perturbation.repeat(batch_image.size(0),1,1,1)),\
                                             inverse_transform(batch_adv_image), adv_title, cln_title, attack_scenario, norm_perturbation, norm_p)
                    self.visualize_attention(images=batch_image, labels=pred, attentions_map=cln_log_attn_1, attentions_attribute=cln_log_attn_2,\
                                             n_top_attr=5, mode="clean")
                    
                    self.visualize_attention(images=batch_adv_image, labels=adv_pred, attentions_map=model.dazle.package_out['A'], attentions_attribute=model.dazle.package_out['A_p'],\
                                             n_top_attr=5, mode="adv", norm=norm_p)




                diff_img = inverse_transform(batch_image) - inverse_transform(batch_adv_image)        
                diff_img = diff_img.abs()
                list_diff_img.append(diff_img) 


            tensor_diff_img = torch.cat(list_diff_img, dim=0)
            avg_per_pixel_perturbation = tensor_diff_img.sum()/tensor_diff_img.size(0)/(224*224)
            if norm_p==2:
                self.logger.add_scalar('avg per pixel_L2/'+attack_scenario, avg_per_pixel_perturbation,norm_perturbation)
                self.logger.add_scalar('true model accuracy_L2/'+attack_scenario, model_accuracy/num_attacked_samples, norm_perturbation)
                self.logger.add_scalar('true model initial accuracy_L2/'+attack_scenario, model_initial_accuracy/num_attacked_samples, norm_perturbation)
            else:
                self.logger.add_scalar('avg per pixel_Linf/'+attack_scenario+'_norm*100', avg_per_pixel_perturbation,norm_perturbation*100)
                self.logger.add_scalar('true model accuracy_Linf/'+attack_scenario+'_norm*100', model_accuracy/num_attacked_samples, norm_perturbation*100)
                self.logger.add_scalar('true model initial accuracy_Linf/'+attack_scenario+'_norm*100', model_initial_accuracy/num_attacked_samples, norm_perturbation*100)

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


            





                
            
            
               
           
           
