# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import models, transforms, utils
from torch.utils.data import DataLoader
from glob import glob
import copy
import os
import sys
from tracker.kalmanBased.qatm.soft_argmax import SoftArgmax2D
from PIL import Image


"""## Using Actual Dataset"""

class TrackingDataset(torch.utils.data.Dataset):
  def __init__(self, data_path, split_length, split_start, transform=None):
    self.split_length = split_length
    self.split_start = split_start
    self.data_path = data_path
    self.transform = transform
    if not self.transform:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ])
    
    self.image_dir_path = os.path.join(data_path, "img")
    self.gt_path = os.path.join(data_path, "groundtruth_rect.txt")

    try:
      self.gt_rects = np.loadtxt(self.gt_path)
    except:
      self.gt_rects = np.loadtxt(self.gt_path, delimiter=",")

    self.image_names = os.listdir(self.image_dir_path)
    self.image_names.sort()
    self.total_seq_len = len(self.image_names)

    # set template
    self.template_rect = self.gt_rects[split_start]
    self.template_path = os.path.join(self.image_dir_path, self.image_names[split_start])
    template_raw = cv2.imread(self.template_path)
    template_raw = Image.fromarray(template_raw)
    template_tensor = transforms.functional.crop(template_raw,
                                                 int(self.template_rect[1]), 
                                                 int(self.template_rect[0]), 
                                                 int(self.template_rect[3]), 
                                                 int(self.template_rect[2]))
    template_tensor = self.transform(template_tensor)
    self.template = template_tensor

  def __getitem__(self, idx):
    self.image_name = self.image_names[self.split_start + idx]
    self.image_raw = cv2.imread(os.path.join(self.image_dir_path, self.image_name))
    self.image = self.transform(self.image_raw)
    center_point = np.zeros(2)
    center_point[0] = self.gt_rects[self.split_start + idx][0] + 0.5*self.gt_rects[self.split_start + idx][2]
    center_point[1] = self.gt_rects[self.split_start + idx][1] + 0.5*self.gt_rects[self.split_start + idx][3]

    return {'image': self.image, 
                'image_raw': self.image_raw, 
                'image_name': self.image_name,
                'template': self.template, 
                'template_name': self.template_path, 
                'template_h': self.template.size()[-2],
                'template_w': self.template.size()[-1],
                'groundtruth': center_point,
                'data_path': self.data_path}

  def __len__(self):
      return min(self.split_length, self.total_seq_len - self.split_start)

def getSingleDatasetsAndLoaders(data_path, batch_size, split_size = 50):
  trackingSets = []
  dataLoaders = []

  dir = data_path
  img_names = os.listdir(os.path.join(dir, "img"))
  seq_len = len(img_names)
  split_start = 0

  while (split_start < seq_len):
    trackSet = TrackingDataset(dir, split_size, split_start)
    trackingSets.append(trackSet)
    loader = DataLoader(trackSet, batch_size=batch_size, shuffle=True)
    dataLoaders.append(loader)
    split_start += split_size

  return trackingSets, dataLoaders

"""### EXTRACT FEATURE"""

def getDatasetsAndLoaders(data_path, batch_size, split_size=50):
    trackingSets = []
    dataLoaders = []

    for dir in glob(os.path.join(data_path, "*")):
        img_names = os.listdir(os.path.join(dir, "img"))
        seq_len = len(img_names)
        split_start = 0

        while (split_start < seq_len):
            trackSet = TrackingDataset(dir, split_size, split_start)
            trackingSets.append(trackSet)
            loader = DataLoader(trackSet, batch_size=batch_size, shuffle=True)
            dataLoaders.append(loader)
            split_start += split_size

    return trackingSets, dataLoaders

class Featex():
    def __init__(self, model, use_cuda):
        self.use_cuda = use_cuda
        self.feature1 = None
        self.feature2 = None
        self.model= copy.deepcopy(model.eval())
        self.model = self.model[:17]
        for param in self.model.parameters():
            param.requires_grad = False
        if self.use_cuda:
            self.model = self.model.cuda()
        self.model[2].register_forward_hook(self.save_feature1)
        self.model[16].register_forward_hook(self.save_feature2)
        
    def save_feature1(self, module, input, output):
        self.feature1 = output.detach()
    
    def save_feature2(self, module, input, output):
        self.feature2 = output.detach()
        
    def __call__(self, input, mode='big'):
        if self.use_cuda:
            input = input.cuda()
        _ = self.model(input)
        # resize feature1 to the same size of feature2
        self.feature1 = F.interpolate(self.feature1, size=(self.feature2.size()[2], self.feature2.size()[3]), mode='bilinear', align_corners=True)
        return torch.cat((self.feature1, self.feature2), dim=1)

"""## QATM"""

class QATM():
    def __init__(self, alpha):
        self.alpha = alpha
        
    def __call__(self, x):
        batch_size, ref_row, ref_col, qry_row, qry_col = x.size()
        x = x.view(batch_size, ref_row*ref_col, qry_row*qry_col)
        xm_ref = x - torch.max(x, dim=1, keepdim=True)[0]
        xm_qry = x - torch.max(x, dim=2, keepdim=True)[0]
        confidence = torch.sqrt(F.softmax(self.alpha*xm_ref, dim=1) * F.softmax(self.alpha * xm_qry, dim=2))
        conf_values, ind3 = torch.topk(confidence, 1)
        ind1, ind2 = torch.meshgrid(torch.arange(batch_size), torch.arange(ref_row*ref_col))
        ind1 = ind1.flatten()
        ind2 = ind2.flatten()
        ind3 = ind3.flatten()
        if x.is_cuda:
            ind1 = ind1.cuda()
            ind2 = ind2.cuda()
        
        values = confidence[ind1, ind2, ind3]
        values = torch.reshape(values, [batch_size, ref_row, ref_col, 1])
        return values
    def compute_output_shape( self, input_shape ):
        bs, H, W, _, _ = input_shape
        return (bs, H, W, 1)

EPSILON = 0.0001

class MyNormLayer():
    def __call__(self, x1, x2):
        bs, _ , H, W = x1.size()
        _, _, h, w = x2.size()
        x1 = x1.view(bs, -1, H*W)
        x2 = x2.view(bs, -1, h*w)
        concat = torch.cat((x1, x2), dim=2)
        x_mean = torch.mean(concat, dim=2, keepdim=True)
        x_std = torch.std(concat, dim=2, keepdim=True)
        x1 = (x1 - x_mean) / (x_std + EPSILON)
        x2 = (x2 - x_mean) / (x_std + EPSILON)
        x1 = x1.view(bs, -1, H, W)
        x2 = x2.view(bs, -1, h, w)
        return [x1, x2]

"""## Corner Head"""

class MaxHead(nn.Module):
  def __init__(self):
    super(MaxHead, self).__init__()

    self.conv1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=(5,5), stride=(1,1), padding=(2,2)),
                                  nn.BatchNorm2d(32),
                                  nn.ReLU())
    self.conv2 = nn.Sequential(nn.Conv2d(32, 16, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                                  nn.BatchNorm2d(16),
                                  nn.ReLU())
    self.conv3 = nn.Sequential(nn.Conv2d(16, 8, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                                  nn.BatchNorm2d(8),
                                  nn.ReLU())
    self.conv4 = nn.Conv2d(8, 1, kernel_size=(1,1), stride=(1,1))
    
    self.max_model = nn.Sequential(self.conv1,
                                   self.conv2,
                                   self.conv3,
                                   self.conv4)
    
  def forward(self, x):
    return self.max_model(x)

"""## Tracking Model"""

QUICK_OPTION = True

class TrackerModel(nn.Module):
  def __init__(self, model, use_cuda):
    super(TrackerModel, self).__init__()
    # make featex untrainable:
    for param in model.parameters():
      param.requires_grad = False
    self.alpha = 25 # change to nn.parameter later
    #self.alpha = nn.Parameter(torch.tensor(25.0, requires_grad=True))
    self.featex = Featex(model, use_cuda)
    self.max_head = MaxHead()
    self.qatm = QATM(self.alpha)
    self.qatm.requires_grad = False
    self.normLayer = MyNormLayer()
    self.normLayer.requires_grad = False
    self.bNormConfMap = nn.BatchNorm2d(1)
    self.use_cuda = use_cuda

    if self.use_cuda:
      self.max_head = self.max_head.cuda()
      self.bNormConfMap = self.bNormConfMap.cuda()

    self.softArgmax2d = SoftArgmax2D()
    
  def set_template(self, template):
    self.T_feat = self.featex(template)
  
  def forward(self, x):
    bs, indim, H, W = x.shape
    I_feat = self.featex(x)
    I_feat_norm, T_feat_norm = self.normLayer(I_feat, torch.cat(bs*[self.T_feat]))

    dist = torch.einsum("xcab,xcde->xabde", I_feat_norm / torch.norm(I_feat_norm, dim=1, keepdim=True), T_feat_norm / torch.norm(T_feat_norm, dim=1, keepdim=True))
    dist.requires_grad = False

    conf_map = self.qatm(dist)
    conf_map_cont = conf_map.permute(0, 3, 1, 2)
    _, dim_template, H_template, W_template = T_feat_norm.shape

    conf_map_cont = torch.log(conf_map_cont)

    _, dim_conf, H_conf, W_conf = conf_map_cont.shape

    if QUICK_OPTION == False:
      conf_map_cont = nn.Upsample(size=(H, W), mode='bilinear', align_corners=True)(conf_map_cont)

    conf_map_cont.requires_grad = False
    heatmap = self.max_head(self.bNormConfMap(conf_map_cont))
    max_point = torch.squeeze(self.softArgmax2d(heatmap), dim=1)

    if QUICK_OPTION == True:
      _, dim, H_heat, W_heat = heatmap.shape
      H_ratio = H / H_heat
      W_ratio = W / W_heat
      max_point[:, 0] = max_point[:, 0] * H_ratio
      max_point[:, 1] = max_point[:, 1] * W_ratio
    
    
    return heatmap.permute(0, 2, 3, 1), max_point


def getValidationLoss(model, trackingSets_val, dataLoaders_val):
  total_loss = 0

  for sequence_num, dataSet in enumerate(trackingSets_val):
    print("*** Sequence: " + str(sequence_num) + " ***\n")
    print("Sequence length: " + str(len(dataSet)) + "\n")
    print("Sequence path: " + dataSet[0]["data_path"] + "\n")
    total_loss_for_seq = 0
    template = torch.unsqueeze(dataSet[0]["template"], 0)
    model.set_template(template)
    dLoader = dataLoaders_val[sequence_num]
    model.eval()

    for batch_count, batch in enumerate(dLoader):
      input = batch["image"].to(device)
      gt = batch["groundtruth"].to(device).float()
      _, result_point = model(batch["image"])
      loss = loss_func(result_point, gt)
      total_loss_for_seq += float(loss.item()) / BATCH_SIZE
      if (batch_count / BATCH_SIZE) % 100 == 0:
        print("Results[0]: ")
        print(result_point[0].detach())
        print("GT[0]: ")
        print(gt[0].detach())

    gc.collect()
    total_loss += total_loss_for_seq
    print("\nTotal loss for sequence (per sample): " + str(total_loss_for_seq) + "\n")
  
  print("Total loss on validation sets: " + str(total_loss))
  return total_loss

