# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse
import cv2
import torch
import numpy as np

from pysot.pysot.utils.bbox import get_axis_aligned_bbox
from pysot.toolkit.datasets import DatasetFactory
from pysot.toolkit.utils.region import vot_overlap, vot_float2str
from pytracking.refine_modules.refine_module import RefineModule
from pytracking.RF_utils import bbox_clip

# base tracker
from arena.LaSOT.common_path_siamrpn import *


###################################################
'''dimp'''
# tracker_param_ = 'dimp50_RF'
# tracker_param_ = 'dimp50_vot_RF'
# tracker_param_ = 'dimp50_vot'
# tracker_param_ = 'dimp50'
tracker_param_ = 'super_dimp'
###################################################
'''dimp'''
tracker_name_ = 'dimp'

from pytracking.evaluation import Tracker
parser = argparse.ArgumentParser(description='Pytracking-RF tracking')
parser.add_argument('--dataset', default= dataset_name_, type=str,
        help='eval one special dataset')
parser.add_argument('--video', default= video_name_, type=str,
        help='eval one special video')
parser.add_argument('--vis', action='store_true',default=False,
        help='whether to visualzie result')
parser.add_argument('--debug', action='store_true',default=False,
        help='whether to debug'),
parser.add_argument('--tracker_name', default= tracker_name_, type=str,
        help='name of tracker for pytracking tracker'),
parser.add_argument('--tracker_param', default= tracker_param_, type=str,
        help='name of param for pytracking tracker')
parser.add_argument('--run_id',type=int, default=1)


args = parser.parse_args()
torch.set_num_threads(1)


def main():
    # create tracker
    tracker_info = Tracker(args.tracker_name, args.tracker_param, None)
    params = tracker_info.get_parameters()
    params.visualization = args.vis
    params.debug = args.debug
    params.visdom_info = {'use_visdom': False, 'server': '127.0.0.1', 'port': 8097}
    tracker = tracker_info.tracker_class(params)

    '''Refinement module'''
    RF_module = RefineModule(refine_path, selector_path, search_factor=sr, input_sz=input_sz)
    model_name = args.tracker_name + '_' + args.tracker_param + '{}-{}'.format(RF_type, selector_path) + '_%d'%(args.run_id)
    model_name = 'LaSOT_gt'

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset, dataset_root=dataset_root_, load_img=False)

    # OPE tracking
    for v_idx, video in enumerate(dataset):
        if os.path.exists(os.path.join(save_dir, args.dataset, model_name, '{}.txt'.format(video.name))):
            continue
        if args.video != '':
            # test one special video
            if video.name != args.video:
                continue
        toc = 0
        pred_bboxes = []
        scores = []
        track_times = []
        for idx, (img, gt_bbox) in enumerate(video):
            cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
            gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
            pred_bboxes.append(gt_bbox_)
            continue

            '''get RGB format image'''
            img_RGB = img[:, :, ::-1].copy()  # BGR --> RGB
            tic = cv2.getTickCount()
            if idx == 0:
                H, W, _ = img.shape
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                '''Initialize'''
                gt_bbox_np = np.array(gt_bbox_)
                gt_bbox_torch = torch.from_numpy(gt_bbox_np.astype(np.float32))
                init_info = {}
                init_info['init_bbox'] = gt_bbox_torch
                _ = tracker.initialize(img_RGB, init_info)
                '''##### initilize refinement module for specific video'''
                RF_module.initialize(img_RGB, np.array(gt_bbox_))

                pred_bbox = gt_bbox_
                scores.append(None)
                pred_bboxes.append(pred_bbox)

            else:
                '''Track'''
                outputs = tracker.track(img_RGB)
                pred_bbox = outputs['target_bbox']
                '''##### refine tracking results #####'''
                pred_bbox = RF_module.refine(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                                             np.array(pred_bbox))
                x1, y1, w, h = pred_bbox.tolist()
                w, h = get_mean_wh(pred_bboxes, w, h)
                '''add boundary and min size limit'''
                x1, y1, x2, y2 = bbox_clip(x1, y1, x1 + w, y1 + h, (H, W))
                w = x2 - x1
                h = y2 - y1
                new_pos = torch.from_numpy(np.array([y1 + h / 2, x1 + w / 2]).astype(np.float32))
                new_target_sz = torch.from_numpy(np.array([h, w]).astype(np.float32))
                new_scale = torch.sqrt(new_target_sz.prod() / tracker.base_target_sz.prod())
                ##### update
                tracker.pos = new_pos.clone()
                tracker.target_sz = new_target_sz
                tracker.target_scale = new_scale

                pred_bboxes.append(pred_bbox)
                # scores.append(outputs['best_score'])
            toc += cv2.getTickCount() - tic
            track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
            if idx == 0:
                cv2.destroyAllWindows()
            if args.vis and idx > 0:
                gt_bbox = list(map(int, gt_bbox))
                pred_bbox = list(map(int, pred_bbox))
                cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                              (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)
                cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                              (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)
                cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow(video.name, img)
                k = cv2.waitKey(0)
                if k == ord('q'):
                    exit()
                elif k == ord('s'):
                    cv2.imwrite(os.path.join(os.environ['HOME'], 'Desktop/demo', video.name+'_{}.jpg'.format(idx)), img)
        # toc /= cv2.getTickFrequency()

        # save results
        model_path = os.path.join(save_dir, args.dataset, model_name)
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        result_path = os.path.join(model_path, '{}.txt'.format(video.name))
        with open(result_path, 'w') as f:
            for x in pred_bboxes:
                f.write(','.join([str(i) for i in x])+'\n')
        print(video.name)
        # print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
        #     v_idx+1, video.name, toc, idx / toc))


def get_mean_wh(boxes, w, h, lam=0.7):
    boxes = boxes[:min(len(boxes), 5)]
    boxes = np.stack(boxes)
    wh = boxes[:, 2:].mean(0)
    return w * lam + wh[0] * (1-lam), h * lam + wh[1] * (1-lam)


if __name__ == '__main__':
    main()
