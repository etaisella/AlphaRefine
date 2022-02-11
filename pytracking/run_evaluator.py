import os
import sys
import argparse
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [14, 8]

from pytracking.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from pytracking.evaluation import Tracker, get_dataset, trackerlist


env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation import get_dataset
from pytracking.evaluation.running import run_dataset
from pytracking.evaluation import Tracker


def run_evaluator(tracker_name, tracker_param, dataset_name='otb', sequence=None, refined_path=None):
    """Run evaluator on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        dataset_name: Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).
        sequence: Sequence number or name.
        refined_path: Path to refined results.
    """
    trackers = []

    trackers.extend(trackerlist(tracker_name, tracker_param, range(0,5), tracker_param))
    dataset = get_dataset(dataset_name)
    eval_data = plot_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'prec'),
                 skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05, exclude_invalid_frames=False)

    print("Evaluation Finished!")


def main():
    parser = argparse.ArgumentParser(description='Run evaluation on tracker and dataset')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of parameter file.')
    parser.add_argument('--dataset_name', type=str, default='otb', help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence number or name.')
    parser.add_argument('--refined_path', type=str, default=None, help='Path to refined results.')

    args = parser.parse_args()

    try:
        seq_name = int(args.sequence)
    except:
        seq_name = args.sequence

    run_evaluator(args.tracker_name, args.tracker_param, args.dataset_name, seq_name, args.refined_path)


if __name__ == '__main__':
    main()
