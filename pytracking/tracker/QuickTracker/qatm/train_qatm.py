from qatm import TrackerModel, getDatasetsAndLoaders
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms, utils
from torch.utils.data import DataLoader
from glob import glob
import torch.optim as optim
import copy
import os
import argparse
import gc
import random

BATCH_SIZE = 12
SPLIT_LENGTH = 32
EPOCHS = 20
LEARNING_RATE = 0.001
SCHEDULER_GAMMA = 0.5



def getValidationLoss(model, trackingSets_val, dataLoaders_val):
  loss_func = nn.MSELoss()
  total_loss = 0
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
  
  
def trainQATM(model_path, dataset_path, val_dataset_path):
    trackingSets, dataLoaders = getDatasetsAndLoaders(dataset_path, BATCH_SIZE, SPLIT_LENGTH)
    trackingSets_val, dataLoaders_val = getDatasetsAndLoaders(val_dataset_path, BATCH_SIZE, SPLIT_LENGTH)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tracker_model = TrackerModel(model=models.vgg19(pretrained=True).features, use_cuda=True)
    optimizer = optim.Adam(lr=LEARNING_RATE, params=tracker_model.parameters())
    loss_func = nn.MSELoss()
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, 3, gamma=SCHEDULER_GAMMA)
    
    training_loss_per_epoch = np.zeros(EPOCHS)
    validation_loss_per_epoch = np.zeros(EPOCHS)
    epochs_finished = 0
    
    for epoch in range(EPOCHS):
        print("*********** EPOCH: " + str(epoch) + " ************\n")
        total_loss = 0
        rand_indices = np.arange(len(trackingSets))
        random.shuffle(rand_indices)
        
        for sequence_num in rand_indices:
            torch.cuda.empty_cache()
            print("*** Sequence: " + str(sequence_num) + " ***\n")
            #print("Sequence length: " + str(len(dataSet)) + "\n")
            print("Sequence path: " + trackingSets[sequence_num][0]["data_path"] + "\n")
            total_loss_for_seq = 0
            template = torch.unsqueeze(trackingSets[sequence_num][0]["template"], 0)
            tracker_model.set_template(template)
            dLoader = dataLoaders[sequence_num]
            tracker_model.train()
        
            for batch_count, batch in enumerate(dLoader):
                optimizer.zero_grad()
                input = batch["image"].to(device)
                gt = batch["groundtruth"].to(device).float()
                _, result = tracker_model(batch["image"])
                loss = loss_func(result, gt)
                loss.backward()
                optimizer.step()
                total_loss_for_seq += loss.item() / BATCH_SIZE
        
            gc.collect()
            total_loss += total_loss_for_seq
            print("\nTotal loss for sequence (per sample): " + str(total_loss_for_seq) + "\n")
        
        training_loss_per_epoch[epoch] = total_loss
        epochs_finished += 1
        scheduler.step()
        print("Evaluating on Validation set:\n")
        validation_loss_per_epoch[epoch] = getValidationLoss(tracker_model, 
                                                            trackingSets_val, 
                                                            dataLoaders_val)
        print("\nTotal training loss for epoch (per sample): " + str(total_loss) + "\n")
        torch.save(tracker_model.state_dict(), model_path)
        
    #torch.save(tracker_model.state_dict(), model_path, _use_new_zipfile_serialization=False)
    plt.plot(training_loss_per_epoch[:epochs_finished])
    plt.plot(validation_loss_per_epoch[:epochs_finished])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()
    
   
  

def main():
    parser = argparse.ArgumentParser(description='Evaluate qatm centerpoint tracker on Dataset')
    parser.add_argument('model_path', type=str, help='Path to centerpoint tracker file')
    parser.add_argument('dataset_path', type=str, help='Path to datast')
    parser.add_argument('val_dataset_path', type=str, help='Path to datast')
    
    args = parser.parse_args()
   
    trainQATM(args.model_path, args.dataset_path, args.val_dataset_path)


if __name__ == '__main__':
    main()