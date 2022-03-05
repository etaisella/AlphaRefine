from qatm import TrackerModel, getSingleDatasetsAndLoaders
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms, utils
from torch.utils.data import DataLoader
from glob import glob
import copy
import os
import argparse


def evaluate_qatm(model_path, dataset_path, sequence):
    SET_NUM = 10
    SAMPLE_IDX = 15
    
    loaded_model = TrackerModel(model=models.vgg19(pretrained=True).features, use_cuda=True)
    loaded_model.load_state_dict(torch.load(model_path))
    loaded_model.eval()

    trackingSets, _ = getSingleDatasetsAndLoaders(dataset_path, 1)


    sample = trackingSets[SET_NUM][SAMPLE_IDX]
    loaded_model.set_template(torch.unsqueeze(sample["template"], dim=0))
    conf_map, bbox = loaded_model(torch.unsqueeze(sample["image"], dim=0))
    conf_map = np.array(conf_map.detach().cpu())
    
    bbox = np.array(bbox.detach().cpu().squeeze()).astype(int)
    image = cv2.imread(os.path.join(sample["data_path"],"img",sample["image_name"]))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.circle(image, (bbox[0], bbox[1]), 10, (255, 0, 0), 4)
    gt = np.array(sample["groundtruth"]).astype(int)
    image = cv2.circle(image, (gt[0], gt[1]), 10, (0, 0, 255), 4)
    
    f, axarr = plt.subplots(1,2)
    axarr[1].imshow(np.squeeze(image))
    axarr[0].imshow(np.squeeze(conf_map))
    plt.show()
    

def main():
    parser = argparse.ArgumentParser(description='Evaluate qatm centerpoint tracker on Dataset')
    parser.add_argument('model_path', type=str, help='Path to centerpoint tracker file')
    parser.add_argument('dataset_path', type=str, help='Path to datast')
    parser.add_argument('--sequence', type=str, default='Woman', help='Sequence number or name.')
    
    args = parser.parse_args()
    
    try:
        seq_name = int(args.sequence)
    except:
        seq_name = args.sequence
    
    evaluate_qatm(args.model_path, args.dataset_path, seq_name)


if __name__ == '__main__':
    main()