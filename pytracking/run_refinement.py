import os
import sys
import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['figure.figsize'] = [14, 8]

from pytracking.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from pytracking.evaluation import Tracker, get_dataset, trackerlist
from pytracking.refine_modules.refine_module import RefineModule


env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation import get_dataset
from pytracking.evaluation.running import run_dataset
from pytracking.evaluation import Tracker

def run_refinement(tracker_name, tracker_param, dataset_name='otb', sequence='Basketball', refined_path='refined'):
    """Run evaluator on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        dataset_name: Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).
        sequence: Sequence number or name.
        refined_path: Path to refined results.
    """
    trackers = []

    tracker_result_path = os.path.join("tracking_results", tracker_name, tracker_param, sequence + '.txt')
    ar_path = 'SEcmnet_ep0040-c.pth.tar'

    # load refinement model and results
    sr = 2.0;
    input_sz = int(128 * sr)  # 2.0 by default
    RF_module = RefineModule(ar_path, 0)
    track_result = np.loadtxt(tracker_result_path, dtype=float, delimiter='\t')

    # get dataset and groundtruth
    dataset = get_dataset(dataset_name)
    data_sequence = None
    for seq in dataset:
        if seq.name == sequence:
            data_sequence = seq

    if data_sequence == None:
        raise ValueError("Bad sequence name received, or none given!")

    gt = data_sequence.ground_truth_rect[0]
    first_image = cv2.cvtColor(cv2.imread(dataset[0].frames[0]), cv2.COLOR_BGR2RGB)

    # initialize refinement model
    RF_module.initialize(first_image, gt)

    refined_results = np.zeros_like(track_result)

    for i, frame in enumerate(data_sequence.frames):
        init_res = track_result[i]
        img = cv2.cvtColor(cv2.imread(frame), cv2.COLOR_BGR2RGB)
        out_dict = RF_module.refine(img, init_res, mode='all', test=True)
        refined_results[i] = out_dict['corner']

    # save results
    tracker_res_refined_dir = os.path.join("tracking_results", tracker_name, tracker_param, "refined")
    if not os.path.exists(tracker_res_refined_dir):
        os.makedirs(tracker_res_refined_dir)
    result_path = os.path.join(tracker_res_refined_dir, sequence + ".txt")
    np.savetxt(result_path, refined_results, delimiter='\t')
    print("Refinement Finished!")


def main():
    parser = argparse.ArgumentParser(description='Run evaluation on tracker and dataset')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of parameter file.')
    parser.add_argument('--dataset_name', type=str, default='otb', help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).')
    parser.add_argument('--sequence', type=str, default='Basketball', help='Sequence number or name.')
    parser.add_argument('--refined_path', type=str, default="refined", help='Refined folder name')

    args = parser.parse_args()

    try:
        seq_name = int(args.sequence)
    except:
        seq_name = args.sequence

    run_refinement(args.tracker_name, args.tracker_param, args.dataset_name, seq_name, args.refined_path)


if __name__ == '__main__':
    main()
