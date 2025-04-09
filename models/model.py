import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):
    return nn.Sequential(
        nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False),
        nn.BatchNorm3d(out_planes)
    )


class disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super(disparityregression, self).__init__()
        self.disp = torch.Tensor(np.reshape(np.array(range(maxdisp)), [1, maxdisp, 1, 1])).cuda() - maxdisp/2

    def forward(self, x):
        out = torch.sum(x*self.disp.data, 1, keepdim=False)
        return out


class CAModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(CAModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.shared = nn.Sequential(
            nn.Conv3d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared(self.avg_pool(x))
        maxout = self.shared(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SAModule(nn.Module):
    def __init__(self):
        super(SAModule, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv3d(out))
        return out


class mmcs(nn.Module):
    def __init__(self, channel):
        super(mmcs, self).__init__()
        self.ca = CAModule(channel)
        self.sa = SAModule()

    def forward(self, x):
        out = self.ca(x) * x
        out = self.sa(out) * out
        return out


class Block(nn.Module):
    def __init__(self, in_places, places, stride=1, downsampling=False, expansion=4):
        super(Block, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv3d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(places),
            nn.ReLU(),
            nn.Conv3d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(places),
            nn.ReLU(),
            nn.Conv3d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(places*self.expansion),
        )
        self.mmcs = mmcs(channel=places*self.expansion)

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(places*self.expansion)
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        out = self.mmcs(out)
 
        if self.downsampling:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)
        return out


class Mul_fuse_3D(nn.Module):
    def __init__(self, input_size=128, hidden_size=256):
        super(Mul_fuse_3D, self).__init__()
        self.conv1 = nn.Conv3d(input_size, hidden_size, kernel_size=3, padding=1, stride=2)
        self.conv2 = nn.Conv3d(hidden_size, hidden_size, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv3d(hidden_size, hidden_size, kernel_size=3, padding=1, stride=2)
        self.conv4 = nn.Conv3d(hidden_size, hidden_size, kernel_size=3, padding=1, stride=1)
        self.conv5 = nn.Sequential(nn.ConvTranspose3d(hidden_size, hidden_size, kernel_size=3, padding=1, output_padding=1, stride=2),
                                   nn.BatchNorm3d(hidden_size)) 
        self.conv6 = nn.Sequential(nn.ConvTranspose3d(hidden_size, input_size, kernel_size=3, padding=1, output_padding=1, stride=2),
                                   nn.BatchNorm3d(input_size)) 
        self.fuse_3D = fuse_3D(64, 64, 128)
        self.fuse_3d1 = fuse_3D(128, 128, 64)

        self.conv7 = nn.Conv3d(128, 128, 3, 1, 1)
        self.conv8 = nn.Conv3d(64, 64, 3, 1, 1)
        self.pool = nn.MaxPool3d(3 ,1, 1)
        self.bn = nn.BatchNorm3d(128)

        self.bn1= nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(3 ,1, 1)

    def forward(self, x, presqu, postsqu):
        x = self.fuse_3D(x)   #128 -> 256
        x = self.bn(x)
        x = torch.relu(self.conv7(x))
        x = self.pool(x)
        
        cls = x[:,:,:1] # B C 1 H W  -> B C H W -> conv2d -> B C 1 H W 
        disp = x[:,:,1:] # 5D B C  D H W
       
        pre = cls * disp
        disp = torch.relu(self.conv1(pre))   # 256 -> 512
        pre = self.conv2(disp)

        if postsqu is not None:
            pre = F.relu(pre+postsqu)
        else:
            pre = F.relu(pre)

        out  = torch.relu(self.conv3(pre))      
        out  = torch.relu(self.conv4(out))
  
        if presqu is not None:
           post = F.relu(self.conv5(out)+presqu)    #  512 -> 512 
        else:
           post = F.relu(self.conv5(out)+pre) 
  
        out = self.conv6(post)   # 512 -> 256


        out = torch.cat((cls, out), dim=2)

        out = self.fuse_3d1(out)
        out = self.bn1(out)
        out = torch.relu(self.conv8(out))
        out = self.pool1(out)
        return out, pre, post



class fuse_3D(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(fuse_3D, self).__init__()
        self.conv1 = nn.Conv3d(input_size, hidden_size, kernel_size=3, padding=1, stride=1)
        self.fuse1 = nn.Conv3d(input_size, hidden_size, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv3d(hidden_size, output_size, kernel_size=3, padding=1, stride=1)
        self.fuse2 = nn.Conv3d(hidden_size, output_size, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        hidden1 = torch.relu(self.conv1(x))
        fuse1 = torch.sigmoid(self.fuse1(x))
        fsue_hidden1 = hidden1 * fuse1

        output = torch.relu(self.conv2(fsue_hidden1))
        fuse2 = torch.sigmoid(self.fuse2(fsue_hidden1))
        final_output = output * fuse2

        return final_output



class fuse_2D(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(fuse_2D, self).__init__()
        self.conv1 = nn.Conv2d(input_size, hidden_size, kernel_size=3, padding=1, stride=1)
        self.fuse1 = nn.Conv2d(input_size, hidden_size, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(hidden_size, output_size, kernel_size=3, padding=1, stride=1)
        self.fuse2 = nn.Conv2d(hidden_size, output_size, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        hidden1 = torch.relu(self.conv1(x))
        fuse1 = torch.sigmoid(self.fuse1(x))
        fuse_hidden1 = hidden1 * fuse1

        output = torch.relu(self.conv2(fuse_hidden1))
        fuse2 = torch.sigmoid(self.fuse2(fuse_hidden1))
        final_output = output * fuse2

        return final_output



def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU())

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

        self.weight = nn.Parameter(torch.Tensor([1]))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out = out+self.weight * out + x

        return out


class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()
        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU(),
                                       nn.MaxPool2d(3, 1, 1))
        self.conv2 = nn.Sequential(nn.ConvTranspose2d(32,64,3,2,1,1,bias=False),
                                   nn.ReLU(),
                                   nn.ConvTranspose2d(64,64,1,1,0,bias=False))
        
        self.conv3 = nn.Sequential(nn.ConvTranspose2d(64,128,3,2,1,1,bias=False),
                                   nn.ReLU(),
                                   nn.ConvTranspose2d(128,128,1,1,0,bias=False))
        
        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 4, 2, 1, 1) 
        self.layer3 = self._make_layer(BasicBlock, 128, 6, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 256, 3, 1, 1, 2)


        self.fuse1 = fuse_2D(64, 512, 512)
        self.fuse2 = fuse_2D(128, 512, 512)
        self.fuse3 = fuse_2D(256, 512, 512)
        
        self.lastconv = self.lastconv = nn.Sequential(nn.Conv2d(512, 128, 3, 1, 1, 1),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(),
                                        nn.Conv2d(128, 32, kernel_size=1, padding=0, stride = 1))

    def forward(self, x):
        conv_x = self.firstconv(x)
        layer1 = self.layer1(conv_x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        output = self.fuse1(layer2) + self.fuse2(layer3) + self.fuse3(layer4)

        output = self.lastconv(output)

        return output
    
    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                            kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)


class cls_extraction(nn.Module):
    def __init__(self):
        super(cls_extraction, self).__init__()

        self.conv = nn.Sequential(nn.Conv2d(3,32,3,2,1),
                                  nn.ReLU(),
                                  nn.Conv2d(32,32,3,1,1),
                                  nn.ReLU(),
                                  nn.Conv2d(32,32,3,2,1),
                                  nn.BatchNorm2d(32),
                                  nn.ReLU(),
                                  nn.MaxPool2d(3, 1, 1),)
        
        self.fuse = fuse_2D(32, 64, 32)
        

    def forward(self, x):

        out = self.conv(x)
        out = self.fuse(out)

        return out


class SSNet(nn.Module):
    def __init__(self, maxdisp=48, mindisp=-48, num_classes=6):
        super(SSNet, self).__init__()
        self.maxdisp = maxdisp
        self.num_classes = num_classes

        self.feature_extraction = feature_extraction()
        self.cls_extraction = cls_extraction()
        
    
        self.fuse = fuse_3D(64, 64, 64)
        self.bn = nn.BatchNorm3d(64)
        self.Mul_fuse_3D1 = Mul_fuse_3D()
        self.Mul_fuse_3D2 = Mul_fuse_3D()
        self.Mul_fuse_3D3 = Mul_fuse_3D()

        self.block1 = Block(in_places=64, places=16)
        self.block2 = Block(in_places=64, places=16)
        self.block3 = Block(in_places=64, places=16)


        self.classif1 = nn.Sequential(convbn_3d(64, 64, 3, 1, 1),
                                nn.ReLU())
        
        self.classif1_1 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                nn.ReLU())
        
        self.classif1_last = nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False)
        
        self.classif2 = nn.Sequential(convbn_3d(64, 64, 3, 1, 1),
                                    nn.ReLU())
        self.classif2_1 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                        nn.ReLU())
        
        self.classif2_last = nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False)
        
        self.classif3 = nn.Sequential(convbn_3d(64, 64, 3, 1, 1),
                                    nn.ReLU())
        self.classif3_1 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                        nn.ReLU())
        
        self.classif3_last = nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False)

        self.fuse1 = fuse_2D(256,128,128)
        self.fuse2 = fuse_2D(128,64,64)
        self.fuse3 = fuse_2D(64,32,32)
        self.weight1 = nn.Parameter(torch.Tensor([1]))
        self.weight2 = nn.Parameter(torch.Tensor([1]))
        self.weight3 = nn.Parameter(torch.Tensor([1]))

        self.conv1 = nn.Conv2d(32,self.num_classes,3,1,1)

        self.conv2 = nn.Sequential(nn.ConvTranspose2d(256,128,3,2,1,1,bias=False),
                                   nn.ReLU(),
                                   nn.ConvTranspose2d(128,128,1,1,0,bias=False))
        
        self.conv3 = nn.Sequential(nn.ConvTranspose2d(128,64,3,2,1,1,bias=False),
                                   nn.ReLU(),
                                   nn.ConvTranspose2d(64,64,1,1,0,bias=False))

        self.firstconv = nn.Sequential(convbn(32, 64, 3, 1, 1, 1),
                        nn.ReLU(),
                        convbn(64, 128, 3, 1, 1, 1),
                        nn.ReLU(),
                        nn.MaxPool2d(3, 1, 1),
                        convbn(128, 256, 3, 1, 1, 1),
                        nn.ReLU(),
                        convbn(256, 256, 3, 1, 1, 1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.MaxPool2d(3, 1, 1))

    def forward(self, left, right):
        left_feature = self.feature_extraction(left)
        right_feature = self.feature_extraction(right)

        lef_cls_first = self.cls_extraction(left).unsqueeze(2)
        rig_cls_first = self.cls_extraction(right).unsqueeze(2)

        cls_first = torch.cat((lef_cls_first, rig_cls_first), dim=1)


        cost = torch.zeros(left_feature.size()[0], left_feature.size()[1]*2, self.maxdisp//4, left_feature.size()[2], left_feature.size()[3]).cuda()
        
        for i in range(-self.maxdisp//8, self.maxdisp//8):
            if i > 0:
                cost[:, :left_feature.size()[1], i + self.maxdisp//8, :, i:] = left_feature[:, :, :, i:]
                cost[:, left_feature.size()[1]:, i + self.maxdisp//8, :, i:] = right_feature[:, :, :, :-i]
            elif i == 0:
                cost[:, :left_feature.size()[1], self.maxdisp//8, :, :] = left_feature
                cost[:, left_feature.size()[1]:, self.maxdisp//8, :, :] = right_feature
            else:
                cost[:, :left_feature.size()[1], i + self.maxdisp//8, :, :i] = left_feature[:, :, :, :i]
                cost[:, left_feature.size()[1]:, i + self.maxdisp//8, :, :i] = right_feature[:, :, :, -i:]
        
        cost = torch.cat((cls_first, cost), dim=2)
        cost = cost.contiguous()


        cost0 = self.fuse(cost)
        cost0 = self.bn(cost0)

        out1, pre1, post1 = self.Mul_fuse_3D1(cost0, None, None)
        out1 = out1+cost0
        out1 = self.block1(out1)

        out2, pre2, post2 = self.Mul_fuse_3D2(out1, pre1, post1) 
        out2 = out2+cost0
        out2 = self.block2(out2)

        out3, pre3, post3 = self.Mul_fuse_3D3(out2, pre1, post2) 
        out3 = out3+cost0
        out3 = self.block3(out3)


        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2

        cost1 = self.classif1_1(cost1)
        cost2 = self.classif2_1(cost2) + cost1
        cost3 = self.classif3_1(cost3) + cost2
        
        cls1 = cost1[:,:,:1]
        cls2 = cost2[:,:,:1]
        cls3 = cost3[:,:,:1]
        cost1 = cost1[:,:,1:]
        cost2 = cost2[:,:,1:]
        cost3 = cost3[:,:,1:]

        
        cost1 = self.classif1_last(cost1)
        cost2 = self.classif2_last(cost2) + cost1
        cost3 = self.classif3_last(cost3) + cost2

        cost1 = F.interpolate(cost1, [self.maxdisp,left.size()[2],left.size()[3]], align_corners=True, mode='trilinear')
        cost2 = F.interpolate(cost2, [self.maxdisp,left.size()[2],left.size()[3]], align_corners=True, mode='trilinear')

        cost1 = torch.squeeze(cost1,1)
        pred1 = F.softmax(cost1,dim=1)
        pred1 = disparityregression(self.maxdisp)(pred1)

        cost2 = torch.squeeze(cost2,1)
        pred2 = F.softmax(cost2,dim=1)
        pred2 = disparityregression(self.maxdisp)(pred2)

        cost3 = F.interpolate(cost3, [self.maxdisp,left.size()[2],left.size()[3]], align_corners=True, mode='trilinear')
        cost3 = torch.squeeze(cost3,1)
        pred3 = F.softmax(cost3,dim=1)
        pred3 = disparityregression(self.maxdisp)(pred3)
 
        cls1 = cls1.squeeze(2)
        cls2 = cls2.squeeze(2)
        cls3 = cls3.squeeze(2)
  
        cls = cls1+cls2+cls3

        cls1 = self.firstconv(cls)
        cls2 = self.conv2(cls1)
        cls3 = self.conv3(cls2)
 
        cls1 = F.interpolate(cls1, [cls2.size()[2], cls2.size()[3]], align_corners=False, mode='bilinear')
        cls2 = self.weight1*self.fuse1(cls1)+cls2
        cls2 = F.interpolate(cls2, [cls3.size()[2], cls3.size()[3]], align_corners=False, mode='bilinear')
        cls3 = self.weight2*self.fuse2(cls2)+cls3
        cls3 = F.interpolate(cls3, [left.size()[2], left.size()[3]], align_corners=False, mode='bilinear')
        cls3 = self.fuse3(cls3)
        cls3 = self.conv1(cls3)
 
        return pred1, pred2, pred3, cls3

