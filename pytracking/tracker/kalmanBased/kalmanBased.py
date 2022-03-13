from pytracking.tracker.base import BaseTracker
import numpy as np
import numpy.matlib as mb
import math
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import time
from fast_histogram import histogram1d
from PIL import Image
import torch
from torchvision import models, transforms, utils
import sys
sys.path
sys.path.append('./qatm/')
from tracker.kalmanBased.qatm.qatm import TrackerModel
print(sys.path)

def compNormHist(I, S):
    X_boundry = I.shape[1] - 1
    Y_boundry = I.shape[0] - 1

    # create normalised window
    bottom_boundry = min(max(0, int(S[1] - S[3])), Y_boundry - 1)
    top_boundry = max(min(Y_boundry, int(S[1] + S[3])), 1)
    left_boundry = min(max(0, int(S[0] - S[2])), X_boundry - 1)
    right_boundry = max(min(X_boundry, int(S[0] + S[2])), 1)

    normalised_window = I[bottom_boundry:top_boundry,
                        left_boundry:right_boundry,
                        :]

    normalised_window = np.around(normalised_window * (15 / 255))
    normalized_window_gray = normalised_window[:, :, 0] + 16 * normalised_window[:, :, 1] + \
                             256 * normalised_window[:, :, 2]

    # create histogram
    q_test = histogram1d(normalized_window_gray.flat, bins=4096, range=(0, 4096))
    q_test = q_test / np.sum(q_test)

    return q_test


def compBatDist(p, q):
    sigma_part = np.sum(np.sqrt(p * q)) * 20
    return math.exp(sigma_part)

class kb_tracker(BaseTracker):

    def predictParticles(self, S_next_tag):
    	S = np.zeros_like(S_next_tag)
    	rand_x = np.random.normal(scale=self.params.x_rand_scale, size=int(S.shape[1])).astype('int8')
    	rand_y = np.random.normal(scale=self.params.y_rand_scale, size=int(S.shape[1])).astype('int8')
    	rand_box_x = np.random.normal(scale=self.params.x_box_rand_scale, size=int(S.shape[1])).astype('int8')
    	rand_box_y = np.random.normal(scale=self.params.y_box_rand_scale, size=int(S.shape[1])).astype('int8')
    	S[0, :] = S_next_tag[0, :] + S_next_tag[4, :] + rand_x[:]
    	S[1, :] = S_next_tag[1, :] + S_next_tag[5, :] + rand_y[:]
    	S[2, :] = np.maximum(S_next_tag[2, :] + rand_box_x[:], self.params.min_box_width)
    	S[3, :] = np.maximum(S_next_tag[3, :] + rand_box_y[:], self.params.min_box_height)
    	S[4, :] = S_next_tag[4, :] + rand_x[:]
    	S[5, :] = S_next_tag[5, :] + rand_y[:]
    	return S

    def sampleParticles(self, S, W):
        N = S.shape[1]
        S_tag = np.zeros_like(S)
        draw = np.random.choice(np.arange(N), N, p=W[0])
        S_tag[:, np.arange(N)] = S[:, draw]
        return S_tag

    def setTemplateFromImage(self, image, bbox):
        template_raw = Image.fromarray(image)
        template_tensor = transforms.functional.crop(template_raw,
                                                     int(bbox[1]),
                                                     int(bbox[0]),
                                                     int(bbox[3]),
                                                     int(bbox[2]))
        template_tensor = self.transform(template_tensor)
        self.template = template_tensor
    
    
    def calculateW(self, I, S, q, centerPoint):
        N = S.shape[1]
        W = np.zeros((1, N))

        # calculate weights

        X_boundry = I.shape[1] - 1
        Y_boundry = I.shape[0] - 1

        # create normalised window
        bottom_boundry = np.clip(S[1] - S[3], 0, Y_boundry - 1).astype(int)
        top_boundry = np.clip(S[1] + S[3], 1, Y_boundry).astype(int)
        left_boundry = np.clip(S[0] - S[2], 0, X_boundry - 1).astype(int)
        right_boundry = np.clip(S[0] + S[2], 1, X_boundry).astype(int)

        p = np.zeros((N, 4096))
        for i in range(N):
            normalised_window = I[bottom_boundry[i]:top_boundry[i],
                            left_boundry[i]:right_boundry[i],
                            :]

            normalised_window = np.around(normalised_window * (15 / 255))
            normalized_window_gray = normalised_window[:, :, 0] + 16 * normalised_window[:, :, 1] + \
                                 256 * normalised_window[:, :, 2]

            # create histogram
            q_test = histogram1d(normalized_window_gray.flat, bins=4096, range=(0, 4096))
            q_test = q_test / np.sum(q_test)
            p[i] = q_test

        sigma_part = np.sum(np.sqrt(p * q), axis=1) * 10
        # distance = compBatDist(p, q)
        W[0] = np.exp(sigma_part)

        # normalise
        W = W / np.sum(W)

        cp_particles = [S[0], S[1]]
        cp_estimated = centerPoint
        dist = (cp_particles[0] - cp_estimated[0])**2 + (cp_particles[1] - cp_estimated[1])**2

        W2 = np.amax(dist) - dist
        if np.amin(dist) > 30:
            print("Distance too big")
            beta = 0

        W2 = W2 / np.sum(W2)

        W = W + W2 * self.params.lambda_param
        W = W / np.sum(W)

        return W
    
    
    def initialize(self, image, info: dict) -> dict:
        self.transform = self.transform = transforms.Compose([transforms.ToTensor(),
                                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                   std=[0.229, 0.224, 0.225],)
                                                              ])

        N = self.params.num_particles

        # initialize particles
        init_box = info['init_bbox']
        s_initial = np.array([init_box[0] + int(init_box[2] / 2), init_box[1] + int(init_box[3] / 2),
                              int(init_box[2] / 2), int(init_box[3] / 2), 0, 0])

        self.S = self.predictParticles(mb.repmat(s_initial, 1, N).reshape((6, N), order='F'))

        # init qatm model and template
        self.setTemplateFromImage(image, init_box)
        self.qatm_model = TrackerModel(model=models.vgg19(pretrained=True).features, use_cuda=True)
        self.qatm_model.load_state_dict(torch.load("tracker/kalmanBased/qatm/model_files/model.pth"))
        self.qatm_model.set_template(torch.unsqueeze(self.template, dim=0))
        self.qatm_model.eval()

        # initialize histogram and weights
        self.q = compNormHist(image, s_initial)
        self.W = self.calculateW(image, self.S, self.q, [init_box[0] + int(init_box[2] / 2), init_box[1] + int(init_box[3] / 2)])

        self.images_processed = 1

        f = open("centerPoint.txt", "w")
        f.close()

        # Time initialization
        tic = time.time()
        out = {'time': time.time() - tic}
        return out
    
    
    def track(self, image, info: dict = None) -> dict:
        S_prev = self.S

        # sample current particles
        S_next_tag = self.sampleParticles(S_prev, self.W)

        # predinct next particles
        self.S = self.predictParticles(S_next_tag)

        # get center point
        image_tensor = self.transform(image)
        _, center_point = self.qatm_model(torch.unsqueeze(image_tensor, dim=0))
        center_point = np.array(center_point.detach().cpu().squeeze()).astype(int)

        # compute particle weights
        self.W = self.calculateW(image, self.S, self.q, center_point)

        # return agrragated particle
        X_mean = np.sum(np.multiply(self.W, self.S[0, :]))
        Y_mean = np.sum(np.multiply(self.W, self.S[1, :]))
        width_mean = np.sum(np.multiply(self.W, self.S[2, :]))
        height_mean = np.sum(np.multiply(self.W, self.S[3, :]))
        output_state = [X_mean - width_mean, Y_mean - height_mean, width_mean * 2, height_mean * 2]
        out = {'target_bbox': output_state}

        # open cp file (delete later)
        f = open("centerPoint.txt", "a")
        f.write("%f %f \n" % (center_point[0], center_point[1]))
        f.close()

        self.images_processed += 1

        # update gt
        if self.images_processed % self.params.gt_update_interval == 0:
            curr_state_histo = compNormHist(image, np.array([X_mean, Y_mean, width_mean, height_mean]))
            weight = (compBatDist(self.q, curr_state_histo) / compBatDist(1, 1))
            updated_q = self.q + weight * curr_state_histo
            norm = np.linalg.norm(updated_q)
            self.q = updated_q / norm

        return out


