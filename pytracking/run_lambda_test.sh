#!/bin/bash

sequence=$1

sudo rm tracking_results/kalmanBased/standard0_5/refined/$sequence.txt
sudo rm tracking_results/kalmanBased/standard_75/refined/$sequence.txt
sudo rm tracking_results/kalmanBased/standard_1/refined/$sequence.txt
sudo rm tracking_results/kalmanBased/standard1_25/refined/$sequence.txt
sudo rm tracking_results/kalmanBased/standard1_5/refined/$sequence.txt

echo Starting tracking...

python run_tracker.py kalmanBased standard0_5 --sequence=$sequence
python run_tracker.py kalmanBased standard_75 --sequence=$sequence
python run_tracker.py kalmanBased standard_1 --sequence=$sequence
python run_tracker.py kalmanBased standard1_25 --sequence=$sequence
python run_tracker.py kalmanBased standard1_5 --sequence=$sequence

echo Starting refinement...

python run_refinement.py kalmanBased standard0_5 --sequence=$sequence --refined_path=refined
python run_refinement.py kalmanBased standard_75 --sequence=$sequence --refined_path=refined
python run_refinement.py kalmanBased standard_1 --sequence=$sequence --refined_path=refined
python run_refinement.py kalmanBased standard1_25 --sequence=$sequence --refined_path=refined
python run_refinement.py kalmanBased standard1_5 --sequence=$sequence --refined_path=refined

echo Script Finished!