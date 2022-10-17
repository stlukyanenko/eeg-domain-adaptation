import torch.nn as nn
import torch
from torch.autograd import Function

class TemporalAttention(nn.Module):
    def __init__(self, in_channels = 1):
        super(TemporalAttention, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(1, 1))

        self.conv_out = nn.Conv2d(in_channels=in_channels, out_channels = 1, kernel_size=(1, 1))
        self.relu = nn.ReLU()


        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):

        B, C, W, T = x.size()
        Q = self.conv1(x) .permute(0,3,1,2).reshape(B, T, 8 * W)
        K = self.conv2(x) .permute(0,1,2,3).reshape(B, 8 * W, T)
        energy = torch.matmul(Q, K)

        max_H, _ = energy.max(dim=2, keepdim=True)
        min_H, _ = energy.min(dim=2, keepdim=True)
        temp_b = (energy - min_H)
        temp_c = (max_H - min_H)+0.00000001
        energy = temp_b / temp_c
        attention = self.softmax(energy)

        attention = torch.unsqueeze(attention, 1) 

       
        out = torch.matmul(attention, x.reshape(B,C,T,W)).reshape(B,C,W,T)
        
        out = self.relu(self.conv_out(out)) + x

        return out

class SpatialAttention(nn.Module):
    def __init__(self, in_channels = 1):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels= 8, kernel_size=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels= 8, kernel_size=(1, 1))

        self.conv_out = nn.Conv2d(in_channels=in_channels, out_channels = 1, kernel_size=(1, 1))
        self.relu = nn.ReLU()

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=2)


    def forward(self, x):

        B, C, W, T = x.size()
        Q = self.conv1(x).permute(0,3,1,2).reshape(B, W, 8 * T)
        K = self.conv2(x).reshape(B, 8 * T, W)
        energy = torch.matmul(Q, K)

        max_H, _ = energy.max(dim=2, keepdim=True)
        min_H, _ = energy.min(dim=2, keepdim=True)
        temp_b = (energy - min_H)
        temp_c = (max_H - min_H)+0.00000001
        energy = temp_b / temp_c
        attention = self.softmax(energy)

        attention = attention.reshape(B, 1, W, W)
        out = torch.matmul(attention, x)

        out = self.relu(self.conv_out(out)) + x 

        return out

class Attention(nn.Module):
    def __init__(self, in_channels = 1):
        super(Attention, self).__init__()
        self.spatial = SpatialAttention(in_channels = in_channels)
        self.temporal = TemporalAttention(in_channels = in_channels)

    def forward(self, x):
        
        out = torch.cat((x, self.spatial(x), self.temporal(x)), dim=1)

        return out

class DoubleAttention(nn.Module):
    def __init__(self, in_channels = 1):
        super(DoubleAttention, self).__init__()
        self.spatial1 = SpatialAttention(in_channels = 1)
        self.temporal1 = TemporalAttention(in_channels = in_channels)

        self.spatial2 = SpatialAttention(in_channels = in_channels)
        self.temporal2 = TemporalAttention(in_channels = in_channels)

    def forward(self, x):

        out = torch.cat((x, self.spatial2(self.spatial1(x)), self.temporal2(self.temporal1(x))), dim=1)

        return out


class ReverseLayer(Function):
    def forward(ctx, x, a):
        ctx.a = a
        return x.view_as(x)

    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.a, None



class DANN(nn.Module):


    def __init__(self, feature_mode = "attention", use_dropout = False):
        '''
        Use feature mods attention, double_attention, concat
        '''

        super(DANN, self).__init__()

        self.feature_mode = feature_mode


        self.feature = nn.Sequential()
        if feature_mode == "attention":   
            self.feature.add_module('attn', Attention())
        elif feature_mode == "double_attention":
            self.feature.add_module('attn', DoubleAttention())
            
            

        self.feature.add_module('f_conv1', nn.Conv2d(3, 40, kernel_size=(1,25)))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(40))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        if use_dropout:
            self.feature.add_module('f_drop1', nn.Dropout2d())

        self.feature.add_module('f_conv2', nn.Conv2d(40, 80, kernel_size=(22,1)))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(80))
        self.feature.add_module('f_relu2', nn.ReLU(True))
        if use_dropout:
            self.feature.add_module('f_drop2', nn.Dropout2d())

        self.feature.add_module('f_avgpool2', nn.AvgPool2d(kernel_size=(1,75), stride = (1, 15)))


        self.feature.add_module('f_conv3', nn.Conv2d(80, 160, kernel_size=(1,69)))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(80))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        if use_dropout:
            self.feature.add_module('f_drop3', nn.Dropout2d())


        self.class_classifier = nn.Sequential()

        self.class_classifier.add_module('c_fc1', nn.Linear(160, 20))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(20))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        if use_dropout:
            self.class_classifier.add_module('c_drop1', nn.Dropout())
    
        self.class_classifier.add_module('c_fc2', nn.Linear(20, 4))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(160, 80))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(80))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        if use_dropout:
            self.domain_classifier.add_module('d_drop1', nn.Dropout())

        self.domain_classifier.add_module('d_fc2', nn.Linear(80, 40))
        self.domain_classifier.add_module('d_bn2', nn.BatchNorm1d(40))
        self.domain_classifier.add_module('d_relu2', nn.ReLU(True))

        if use_dropout:
            self.domain_classifier.add_module('d_drop2', nn.Dropout())

        self.domain_classifier.add_module('d_fc3', nn.Linear(40, 9))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha):

        if self.feature_mode == "concat":
            feature_data = torch.stack([input_data,input_data,input_data], axis = 1)
        else:
            feature_data = torch.unsqueeze(input_data, 1)
        
        feature = self.feature(feature_data)
        feature = feature.view(-1, 160)
        reverse_feature = ReverseLayer.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output






