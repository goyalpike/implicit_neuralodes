#!/bin/sh
# Training implicit network + NeuralODE for Irregular sampling

########################################################
################ Batch time 2 ##########################
########################################################

mkdir -p ./Results/Linear2D_irregular/

python Linear_IrregularSampling.py --num_epochs 10000 --noise_level 10e-2 --params_datafidelity 0.1  --logging_root './Results' >> ./Results/Linear2D_irregular/Linear_Irregular_bs2.txt && echo "Done with Linear irregular"

python Linear_IrregularSampling.py --num_epochs 2000 --noise_level 20e-2 --params_datafidelity 0.1  --logging_root './Results' >> ./Results/Linear2D_irregular/Linear_Irregular_bs2.txt && echo "Done with Linear irregular"
