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


def run_evaluator():
    """Run evaluator on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        dataset_name: Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).
        sequence: Sequence number or name.
        refined_path: Path to refined results.
    """
    trackers = []

    trackers.extend(trackerlist('kalmanBased', os.path.join('standard_0_5', "refined"), range(0, 5),
                                        "QuickTracker Lambda = 0.5"))
    trackers.extend(trackerlist('kalmanBased', os.path.join('standard_75', "refined"), range(0, 5),
                                        "QuickTracker Lambda = 0.75"))
    trackers.extend(trackerlist('kalmanBased', os.path.join('standard_1', "refined"), range(0, 5),
                                        "QuickTracker Lambda = 1"))
    trackers.extend(trackerlist('kalmanBased', os.path.join('standard_1_25', "refined"), range(0, 5),
                                        "QuickTracker Lambda = 1.25"))
    trackers.extend(trackerlist('kalmanBased', os.path.join('standard_1_5', "refined"), range(0, 5),
                                        "QuickTracker Lambda = 1.5"))                                        

    dataset = get_dataset('otb')
    
    eval_data = plot_results(trackers, dataset, 'otb' + 'refined', merge_results=True, plot_types=('success', 'prec'),
                 skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05, exclude_invalid_frames=False)

    print("Evaluation Finished!")


def main():
    run_evaluator()


if __name__ == '__main__':
    main()