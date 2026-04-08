# -*- coding: utf-8 -*-
import torch
from torch import nn, cat
from torch.nn.functional import dropout
import SMConv
import numpy as np
import torch.nn.functional as F



class droplayer(nn.Module):
	def __init__(self, channel_num=1, thr = 0.3,training=True):
		super(droplayer, self).__init__()
		self.channel_num = channel_num
		self.threshold = thr
		self.training = training
	def forward(self, x):
		if self.training:
			r = torch.rand(x.shape[0],self.channel_num,1,1,1).cuda()
			r[r<self.threshold] = 0
			r[r>=self.threshold] = 1
			r = r*self.channel_num/(r.sum()+0.01)
			return x*r
		else:
			return x
	

class ITFFM(nn.Module):
	def __init__(self, in_ch, out_ch,kernel_size,padding,stride,scale=0.1):
		super(ITFFM, self).__init__()
		self.conv_c = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=3,stride=1,padding=1)
		self.conv_x = SMConv.DeformConv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride, padding=padding,scale=scale)
		self.conv_f = nn.Conv3d(in_channels=2*out_ch, out_channels=out_ch, kernel_size=3,stride=1,padding=1)
		self.In_c = nn.InstanceNorm3d(out_ch)
		self.In_x = nn.InstanceNorm3d(out_ch)
		self.In_f = nn.InstanceNorm3d(out_ch)
		self.LeakyReLU_c = nn.LeakyReLU(inplace=True)
		self.LeakyReLU_x = nn.LeakyReLU(inplace=True)
		self.LeakyReLU_f = nn.LeakyReLU(inplace=True)

		self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=1)
		self.In = nn.InstanceNorm3d(out_ch)
		self.dropout = droplayer(channel_num=out_ch,thr=0.1,training=True)


	def forward(self, x, angle):
		input = x

		conv_c = self.conv_c(x)
		conv_c = self.In_c(conv_c)
		conv_c = self.LeakyReLU_c(conv_c)

		conv_x = self.conv_x(x, angle)
		conv_x = self.In_x(conv_x)
		conv_x = self.LeakyReLU_x(conv_x)
		conv_x = self.dropout(conv_x)

		x = self.conv_f(torch.cat([conv_c,conv_x],dim=1))
		x = self.In_f(x)

		input = self.conv1(input)
		input = self.In(input)

		return self.LeakyReLU_f(x + input)


class SingleConv(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(SingleConv, self).__init__()
		self.conv = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1)
		self.In = nn.InstanceNorm3d(out_ch)
		self.LeakyReLU = nn.LeakyReLU(inplace=True)
		self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=1)
		self.In1 = nn.InstanceNorm3d(out_ch)

	def forward(self, x):
		input = x.clone()
		x = self.conv(x)
		x = self.In(x)

		input = self.conv1(input)
		input = self.In1(input)

		return self.LeakyReLU(x+input)
	
class CLM(nn.Module):
	def __init__(self, in_ch,out_ch):
		super(CLM, self).__init__()
		self.conv_angle_0 = nn.Conv3d(in_ch,out_ch,kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1),bias=False)
		self.conv_angle_1 = nn.Conv3d(out_ch,2,kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1),bias=False)
		self.LeakyReLU = nn.LeakyReLU(inplace=True)

		self.conv_angle_1.weight.data.zero_()


	def forward(self, x):
		x_0 = self.conv_angle_0(x)
		x_0 = self.LeakyReLU(x_0)

		angle = self.conv_angle_1(x_0)
		angle = torch.sigmoid(angle)

		# 映射到球坐标范围
		# theta = (angle[:, 0] + 1) * 0.5 * torch.pi    # [0, π]
		# phi = (angle[:, 1] + 1) * torch.pi            # [0, 2π]

		theta = angle[:, 0] * torch.pi    # [0, π]
		phi = angle[:, 1] * 2 * torch.pi            # [0, 2π]

		# 转换为方向余弦 - 现在可以表示所有方向
		cos_X = torch.sin(theta) * torch.cos(phi)
		cos_Y = torch.sin(theta) * torch.sin(phi)
		cos_Z = torch.cos(theta)

		return torch.stack([cos_Z, cos_Y, cos_X], dim=1),x_0,theta.unsqueeze(1),phi.unsqueeze(1)

class CGSC_Net(nn.Module):
	def __init__(self, n_channels, number,training):
		super(CGSC_Net, self).__init__()
		self.LeakyReLU = nn.LeakyReLU(inplace=True)
		self.number = number 

		self.conv1_1 = ITFFM(n_channels,self.number,9,4,1,scale=0.1,training=training)
		self.conv1_2 = ITFFM(self.number,self.number,9,4,1,scale=0.1,training=training)
		self.conv_angle1 = CLM(n_channels,self.number)
		self.conv_angle2 = CLM(1*self.number,1*self.number)
		

		self.conv2_1 = ITFFM(1*self.number,2*self.number,9,4,1,scale=0.1,training=training)
		self.conv2_2 = ITFFM(2*self.number,2*self.number,9,4,1,scale=0.1,training=training)
		self.conv_angle3 = CLM(1*self.number,2*self.number)
		self.conv_angle4 = CLM(2*self.number,2*self.number)

		self.conv3_1 = ITFFM(2*self.number,4*self.number,9,4,1,scale=0.1,training=training)
		self.conv3_2 = ITFFM(4*self.number,4*self.number,9,4,1,scale=0.1,training=training)
		self.conv_angle5 = CLM(2*self.number,4*self.number)
		self.conv_angle6 = CLM(4*self.number,4*self.number)

		self.conv4_1 = ITFFM(4*self.number,8*self.number,9,4,1,scale=0.1,training=training)
		self.conv4_2 = ITFFM(8*self.number,8*self.number,9,4,1,scale=0.1,training=training)
		self.conv_angle7 = CLM(4*self.number,8*self.number)
		self.conv_angle8 = CLM(8*self.number,8*self.number)

		self.conv5_1 = ITFFM(12*self.number,4*self.number,9,4,1,scale=1,training=training)
		self.conv5_2 = ITFFM(4*self.number,4*self.number,9,4,1,scale=1,training=training)
		self.conv_angle9 = CLM(12*self.number,4*self.number)
		self.conv_angle10 = CLM(4*self.number,4*self.number)

		self.conv6_1 = ITFFM(6*self.number,2*self.number,9,4,1,scale=1,training=training)
		self.conv6_2 = ITFFM(2*self.number,2*self.number,9,4,1,scale=1,training=training)
		self.conv_angle11 = CLM(6*self.number,2*self.number)
		self.conv_angle12 = CLM(2*self.number,2*self.number)

		self.conv7_1 = SingleConv(3*self.number,self.number)
		self.conv7_2 = SingleConv(self.number,self.number)

		########################################################################################3
		
		self.conv_angle13 = CLM(3*self.number,1*self.number)
		self.conv_angle14 = CLM(1*self.number,1*self.number)

		self.out_conv = nn.Conv3d(self.number, 1, 1)
		self.maxpooling = nn.MaxPool3d(2)
		
		self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
		self.up2 = nn.Upsample(scale_factor=2)
		self.up4 = nn.Upsample(scale_factor=4)
		self.up8 = nn.Upsample(scale_factor=8)
		self.sigmoid = nn.Sigmoid()

	


	def forward(self, x):
		
		_,x1_1,theta1,phi1 = self.conv_angle1(x)
		conv_angle2,x1_2,theta2,phi2 = self.conv_angle2(x1_1)
		conv1_1 = self.conv1_1(x, conv_angle2)
		conv1_2 = self.conv1_2(conv1_1,conv_angle2)
		x = self.maxpooling(conv1_2)
		x1_2_down = self.maxpooling(x1_2)

		_,x2_1,_,_ = self.conv_angle3(x1_2_down)	
		conv_angle4,x2_2,theta4,phi4 = self.conv_angle4(x2_1)
		conv2_1 = self.conv2_1(x, conv_angle4)
		conv2_2 = self.conv2_2(conv2_1, conv_angle4)
		x = self.maxpooling(conv2_2)
		x2_2_down = self.maxpooling(x2_2)

		_,x3_1,_,_ = self.conv_angle5(x2_2_down)
		conv_angle6,x3_2,theta6,phi6 = self.conv_angle6(x3_1)
		conv3_1 = self.conv3_1(x, conv_angle6)
		conv3_2 = self.conv3_2(conv3_1, conv_angle6)
		x = self.maxpooling(conv3_2)
		x3_2_down = self.maxpooling(x3_2)

		_,x4_1,_,_ = self.conv_angle7(x3_2_down)
		conv_angle8,x4_2,theta8,phi8 = self.conv_angle8(x4_1)
		conv4_1 = self.conv4_1(x,conv_angle8)
		conv4_2 = self.conv4_2(conv4_1,conv_angle8)
		x = self.up(conv4_2)
		x4_2_up = self.up(x4_2)

		x = torch.cat([x,conv3_2],dim=1)
		_,x5_1,_,_ = self.conv_angle9(torch.cat([x4_2_up,x3_2],dim=1))
		conv_angle10,x5_2,theta10,phi10 = self.conv_angle10(x5_1)
		conv5_1 = self.conv5_1(x, conv_angle10)
		conv5_2 = self.conv5_2(conv5_1, conv_angle10)
		x = self.up(conv5_2)
		x5_2_up = self.up(x5_2)

		x = torch.cat([x,conv2_2],dim=1)
		_,x6_1,_,_ = self.conv_angle11(torch.cat([x5_2_up,x2_2],dim=1))
		conv_angle12,x6_2,theta12,phi12 = self.conv_angle12(x6_1)
		conv6_1 = self.conv6_1(x, conv_angle12)
		conv6_2 = self.conv6_2(conv6_1, conv_angle12)
		x = self.up(conv6_2)
		x6_2_up = self.up(x6_2)
		
		x = torch.cat([x,conv1_2],dim=1)
		conv_angle13,x7_1,theta13,phi13 = self.conv_angle13(torch.cat([x6_2_up,x1_2],dim=1))
		conv_angle14,_,theta14,phi14 = self.conv_angle14(x7_1)
		conv7_1 = self.conv7_1(x)
		conv7_2 = self.conv7_2(conv7_1)

		x = self.out_conv(conv7_2)
		out = self.sigmoid(x)
		
		return out,theta14,phi14
