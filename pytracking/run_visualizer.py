import os
import sys
import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
#plt.rcParams['figure.figsize'] = [14, 8]

from pytracking.evaluation import Tracker, get_dataset, trackerlist

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation import get_dataset
from pytracking.evaluation.running import run_dataset
from pytracking.evaluation import Tracker


def get_ground_truth(dataset_name, sequence):
    if dataset_name == 'otb':
        if sequence == 'Basketball':
            gt_path = os.path.join(sequence, 'groundtruth_rect.txt')
        else:
            raise ValueError('Unsupported Sequence!')
    else:
        raise ValueError('Unsupported Dataset!')

    gt = np.loadtxt(gt_path, dtype=np.float, delimiter=',')
    return gt[0]


def run_visualizer(tracker_name, tracker_param, dataset_name='otb', sequence='Basketball', refined_path=None, show_gui=1):
    """Run evaluator on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        dataset_name: Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).
        sequence: Sequence number or name.
        refined_path: Path to refined results.
    """

    # get tracker results
    tracker_result_path = os.path.join("tracking_results", tracker_name, tracker_param, sequence + '.txt')
    track_result = np.loadtxt(tracker_result_path, dtype=float, delimiter='\t')

    # get refined results
    if refined_path != None:
        refined_result_path = os.path.join("tracking_results", tracker_name, tracker_param, refined_path, sequence + '.txt')
        refined_result = np.loadtxt(refined_result_path, dtype=float, delimiter='\t')

    # get dataset
    dataset = get_dataset(dataset_name)
    data_sequence = None
    for seq in dataset:
        if seq.name == sequence:
            data_sequence = seq

    result_clip = []

    if tracker_name == 'kalmanBased':
        text_scale = 0.7
    else:
        text_scale = 1

    for i, frame in enumerate(data_sequence.frames):
        img = cv2.imread(frame)

        cv2.rectangle(img, (0, 0), (400, 100), (0, 0, 0), thickness=-1)
        cv2.rectangle(img, (0, 0), (400, 100), (255, 255, 255), thickness=1)

        # visualize ground truth
        gt_bbox = list(map(int, data_sequence.ground_truth_rect[i]))
        cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                      (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]), (0, 255, 255), 2)
        cv2.putText(img, 'Ground Truth', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 255), thickness=2)

        # visualize tracker result
        tracker_bbox = list(map(int, track_result[i]))
        cv2.rectangle(img, (tracker_bbox[0], tracker_bbox[1]),
                      (tracker_bbox[0] + tracker_bbox[2], tracker_bbox[1] + tracker_bbox[3]), (0, 0, 255), 2)
        cv2.putText(img, tracker_name + ' Tracker', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 255), thickness=2)

        # if refined path exists add refined results to video
        if refined_path != None:
            refined_bbox = list(map(int, refined_result[i]))
            cv2.rectangle(img, (refined_bbox[0], refined_bbox[1]),
                          (refined_bbox[0] + refined_bbox[2], refined_bbox[1] + refined_bbox[3]), (255, 0, 0), 2)
            cv2.putText(img, 'Refined ' + tracker_name + ' Tracker', (20, 90), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 0, 0), thickness=2)


        result_clip.append(img)

    # save results
    visualizer_dir = os.path.join("visualizer_results", tracker_name, tracker_param)
    if not os.path.exists(visualizer_dir):
        os.makedirs(visualizer_dir)
    result_path = os.path.join(visualizer_dir, sequence + ".avi")
    H, W, _ = img.shape
    out = cv2.VideoWriter(result_path, cv2.VideoWriter_fourcc(*'XVID'), 20, (W, H))

    for img in result_clip:
        img = cv2.resize(img, (W, H))
        '''
        cv2.imshow('', img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            exit(0)
        '''
        out.write(img)
    out.release()

    print("Visualizer Finished!")


def main():
    parser = argparse.ArgumentParser(description='Run evaluation on tracker and dataset')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of parameter file.')
    parser.add_argument('--dataset_name', type=str, default='otb', help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).')
    parser.add_argument('--sequence', type=str, default='Basketball', help='Sequence number or name.')
    parser.add_argument('--refined_path', type=str, default=None, help='Refined folder name')
    parser.add_argument('--show_gui', type=int, default=1, help='Run ID')


    args = parser.parse_args()

    try:
        seq_name = int(args.sequence)
    except:
        seq_name = args.sequence

    run_visualizer(args.tracker_name, args.tracker_param, args.dataset_name, seq_name, args.refined_path, args.show_gui)


if __name__ == '__main__':
    main()
