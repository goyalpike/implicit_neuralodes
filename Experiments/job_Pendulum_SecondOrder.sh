#!/bin/sh
# Training implicit network + NeuralODE + SecondOrder 

mkdir -p ./Results/Pendulum/
mkdir -p ./Results/Pendulum/

########################################################
################ Batch time 3 ##########################
########################################################
python Pendulum_SecondOrder_Implicit_SNODE.py --num_epochs 10000 --noise_level 5e-2  --batch_time 3 --params_datafidelity 1.0 --logging_root './Results' > ./Results/Pendulum/Pendulum_Implicit_NeuralODE_bt3.txt && echo "Done with Implicit_NN with noise 5e-2"

python Pendulum_SecondOrder_Implicit_SNODE.py --num_epochs 1000 --noise_level 2e-1  --batch_time 3 --params_datafidelity 0.1 --logging_root './Results' >> ./Results/Pendulum/Pendulum_Implicit_NeuralODE_bt3.txt && echo "Done with Implicit_NN with noise 2e-1"



# Training only NeuralODE + SecondOrder

########################################################
################ Batch time 3 ##########################
########################################################

python Pendulum_SecondOrder_SNODE.py --num_epochs 1000 --noise_level 5e-2  --batch_time 3  --logging_root './Results' > ./Results/Pendulum/Pendulum_NeuralODE_bt3.txt && echo "Done with NeuralODE with noise 5e-2"

python Pendulum_SecondOrder_SNODE.py --num_epochs 1000 --noise_level 2e-1  --batch_time 3  --logging_root './Results' >> ./Results/Pendulum/Pendulum_NeuralODE_bt3.txt && echo "Done with NeuralODE with noise 2e-1"
