#!/bin/bash

tracker=$1
parameter=$2
sequence=$3
refine_path=$4

sudo rm tracking_results/$tracker/$parameter/$refine_path/$sequence.txt

echo Starting tracking...

python run_tracker.py $tracker $parameter --sequence=$sequence

echo Starting refinement...

python run_refinement.py $tracker $parameter --sequence=$sequence --refined_path=$refined_path

echo Starting visualization...

python run_visualizer.py $tracker $parameter --sequence=$sequence --refined_path=$refined_path

echo Script Finished!
