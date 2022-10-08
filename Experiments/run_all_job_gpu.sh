#!/bin/bash

# YOUR CODE
eval "$(conda shell.bash hook)"

conda activate Implicit_NODE # Vitual enviornment

echo  "Running pendulum example "
sh ./job_Pendulum_SecondOrder.sh

echo  "Running Linear Irregular"
sh ./job_Linear_Irregular.sh

echo  "Running Cubic2D"
sh ./job_Cubic2D.sh

echo "Running quantitative analysis"
python Cubic2D_QualitativeAnalysis.py


