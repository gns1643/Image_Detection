import torch.nn as nn
import torch.nn.functional as F
import torch
import functools
from torchvision import models
from torch.autograd import Variable
import numpy as np
import math

# 기본 정규화 레이어 (필요에 따라 변경 가능)
_norm_layer = nn.BatchNorm2d

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
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

        # Norm 레이어가 뒤에 오면 bias=False가 일반적입니다.
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=False),
                       norm_layer(dim),
                       activation]
        
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
            
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=False),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class GlobalGenerator2(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=2, n_blocks=9, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect', use_sig=False, n_UPsampling=0, use_dropout=True):
        assert(n_blocks >= 0)
        super(GlobalGenerator2, self).__init__()        
        activation = nn.ReLU(True)        

        mult = 1
        # Initial convolution block
        model = [nn.ReflectionPad2d(3), 
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=False), 
                 norm_layer(ngf), 
                 activation]

        ### downsample
        for i in range(n_downsampling):
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=False),
                      norm_layer(ngf * mult * 2), activation]
            mult *= 2

        ### resnet blocks
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer, use_dropout=use_dropout)]

        if n_UPsampling <= 0:
            n_UPsampling = n_downsampling

        ### upsample         
        for i in range(n_UPsampling):
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                       norm_layer(int(ngf * mult / 2)), activation]
            mult = int(mult / 2)

        # Output layer
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if use_sig:
            model += [nn.Sigmoid()]
        else:      
            model += [nn.Tanh()]        
            
        self.model = nn.Sequential(*model)
            
    def forward(self, input, cond=None):
        return self.model(input)

class APDrawingGenerator(GlobalGenerator2):
    def __init__(self, input_nc=3, output_nc=1, ngf=64):
        # 가중치 파일(150_net_gen.pt) 구조: n_downsampling=2, n_blocks=9, use_dropout=True
        super(APDrawingGenerator, self).__init__(input_nc, output_nc, ngf, n_downsampling=2, n_blocks=9, use_dropout=True)

    def forward(self, x):
        # GlobalGenerator2의 forward가 self.model(input)을 수행하므로 그대로 사용
        return super(APDrawingGenerator, self).forward(x)


# --- 이하 기존 코드 유지 (필요한 경우에만 사용) ---

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        # 여기서는 전역 _norm_layer 대신 InstanceNorm2d 사용 (기존 코드 호환성)
        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)
                        ]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9, sigmoid=True):
        super(Generator, self).__init__()
        model0 = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]
        self.model0 = nn.Sequential(*model0)
        model1 = []
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model1 += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2
        self.model1 = nn.Sequential(*model1)
        model2 = []
        for _ in range(n_residual_blocks):
            model2 += [ResidualBlock(in_features)]
        self.model2 = nn.Sequential(*model2)
        model3 = []
        out_features = in_features//2
        for _ in range(2):
            model3 += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2
        self.model3 = nn.Sequential(*model3)
        model4 = [  nn.ReflectionPad2d(3),
                        nn.Conv2d(64, output_nc, 7)]
        if sigmoid:
            model4 += [nn.Sigmoid()]
        self.model4 = nn.Sequential(*model4)

    def forward(self, x, cond=None):
        out = self.model0(x)
        out = self.model1(out)
        out = self.model2(out)
        out = self.model3(out)
        out = self.model4(out)
        return out

class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, use_sigmoid=True):
        super(UnetGenerator, self).__init__()
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    def forward(self, input):
        return self.model(input)

class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, use_sigmoid=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        use_bias = (norm_layer == nn.InstanceNorm2d)
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            if use_sigmoid:
                up = [uprelu, upconv, nn.Sigmoid()]
            else:
                up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
        self.model = nn.Sequential(*model)
    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

class APDrawingLocalGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=1, ngf=32, n_blocks=3):
        super(APDrawingLocalGenerator, self).__init__()
        self.model = Generator(input_nc, output_nc, n_blocks, sigmoid=True)
    def forward(self, x):
        return self.model(x)

class InceptionV3(nn.Module):
    def __init__(self, num_classes, isTrain, use_aux=True, pretrain=False, freeze=True, every_feat=False):
        super(InceptionV3, self).__init__()
        self.every_feat = every_feat
        self.model_ft = models.inception_v3(pretrained=pretrain)
        stop = 0
        if freeze and pretrain:
            for child in self.model_ft.children():
                if stop < 17:
                    for param in child.parameters():
                        param.requires_grad = False
                stop += 1
        num_ftrs = self.model_ft.AuxLogits.fc.in_features
        self.model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs,num_classes)
        self.model_ft.input_size = 299
        self.isTrain = isTrain
        self.use_aux = use_aux
        if self.isTrain:
            self.model_ft.train()
        else:
            self.model_ft.eval()
    def forward(self, x, cond=None, catch_gates=False):
        x = self.model_ft.Conv2d_1a_3x3(x)
        x = self.model_ft.Conv2d_2a_3x3(x)
        x = self.model_ft.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.model_ft.Conv2d_3b_1x1(x)
        x = self.model_ft.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.model_ft.Mixed_5b(x)
        feat1 = x
        x = self.model_ft.Mixed_5c(x)
        feat11 = x
        x = self.model_ft.Mixed_5d(x)
        feat12 = x
        x = self.model_ft.Mixed_6a(x)
        feat2 = x
        x = self.model_ft.Mixed_6b(x)
        feat21 = x
        x = self.model_ft.Mixed_6c(x)
        feat22 = x
        x = self.model_ft.Mixed_6d(x)
        feat23 = x
        x = self.model_ft.Mixed_6e(x)
        feat3 = x
        aux_defined = self.isTrain and self.use_aux
        if aux_defined:
            aux = self.model_ft.AuxLogits(x)
        else:
            aux = None
        x = self.model_ft.Mixed_7a(x)
        x = self.model_ft.Mixed_7b(x)
        x = self.model_ft.Mixed_7c(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        feats = F.dropout(x, training=self.isTrain)
        x = torch.flatten(feats, 1)
        x = self.model_ft.fc(x)
        if self.every_feat:
            return x, feat21
        return x, aux
