# Training implicit network + NeuralODE for Cubic example

mkdir -p ./Results/Cubic2D/

########################################################
################ Batch time 4 ##########################
########################################################
python Cubic2D_Implicit_NeuralODE.py --num_epochs 10000 --noise_level 0.0  --batch_time 4 --params_datafidelity 1.0 --logging_root './Results' > ./Results/Cubic2D/Cubic2D_Implicit_NeuralODE_bt4.txt && echo "Done with Implicit_NN with noise 0.0"

python Cubic2D_Implicit_NeuralODE.py --num_epochs 10000 --noise_level 1e-2 --batch_time 4 --params_datafidelity 1.0 --logging_root './Results' >> ./Results/Cubic2D/Cubic2D_Implicit_NeuralODE_bt4.txt && echo "Done with Implicit_NN with noise 1e-2"

python Cubic2D_Implicit_NeuralODE.py --num_epochs 10000 --noise_level 5e-2  --batch_time 4 --params_datafidelity 1.0 --logging_root './Results' >> ./Results/Cubic2D/Cubic2D_Implicit_NeuralODE_bt4.txt && echo "Done with Implicit_NN with noise 5e-2"

python Cubic2D_Implicit_NeuralODE.py --num_epochs 10000 --noise_level 1e-1  --batch_time 4 --params_datafidelity 0.5 --logging_root './Results' >> ./Results/Cubic2D/Cubic2D_Implicit_NeuralODE_bt4.txt && echo "Done with Implicit_NN with noise 1e-1"

python Cubic2D_Implicit_NeuralODE.py --num_epochs 10000 --noise_level 2e-1  --batch_time 4 --params_datafidelity 0.5 --logging_root './Results' >> ./Results/Cubic2D/Cubic2D_Implicit_NeuralODE_bt4.txt && echo "Done with Implicit_NN with noise 2e-1"

python Cubic2D_Implicit_NeuralODE.py --num_epochs 10000 --noise_level 3e-1  --batch_time 4 --params_datafidelity 0.5 --logging_root './Results' >> ./Results/Cubic2D/Cubic2D_Implicit_NeuralODE_bt4.txt && echo "Done with Implicit_NN with noise 3e-1"

python Cubic2D_Implicit_NeuralODE.py --num_epochs 10000 --noise_level 4e-1  --batch_time 4 --params_datafidelity 0.2 --logging_root './Results' >> ./Results/Cubic2D/Cubic2D_Implicit_NeuralODE_bt4.txt && echo "Done with Implicit_NN with noise 4e-1"

python Cubic2D_Implicit_NeuralODE.py --num_epochs 10000 --noise_level 5e-1  --batch_time 4 --params_datafidelity 0.2 --logging_root './Results' >> ./Results/Cubic2D/Cubic2D_Implicit_NeuralODE_bt4.txt && echo "Done with Implicit_NN with noise 5e-1"


###############################################
###############################################

# Training NeuralODE for Cubic example

########################################################
################ Batch time 4 ##########################
########################################################
python Cubic2D_NeuralODE.py --num_epochs 10000 --noise_level 0.0  --batch_time 4  --logging_root './Results' > ./Results/Cubic2D/Cubic2D_NeuralODE_bt4.txt && echo "Done with NeuralODE with noise 0.0"

python Cubic2D_NeuralODE.py --num_epochs 10000 --noise_level 1e-2 --batch_time 4  --logging_root './Results' >> ./Results/Cubic2D/Cubic2D_NeuralODE_bt4.txt && echo "Done with NeuralODE with noise 1e-2"

python Cubic2D_NeuralODE.py --num_epochs 1000 --noise_level 5e-2  --batch_time 4  --logging_root './Results' >> ./Results/Cubic2D/Cubic2D_NeuralODE_bt4.txt && echo "Done with NeuralODE with noise 5e-2"

python Cubic2D_NeuralODE.py --num_epochs 1000 --noise_level 1e-1  --batch_time 4  --logging_root './Results' >> ./Results/Cubic2D/Cubic2D_NeuralODE_bt4.txt && echo "Done with NeuralODE with noise 1e-1"

python Cubic2D_NeuralODE.py --num_epochs 1000 --noise_level 2e-1  --batch_time 4  --logging_root './Results' >> ./Results/Cubic2D/Cubic2D_NeuralODE_bt4.txt && echo "Done with NeuralODE with noise 2e-1"

python Cubic2D_NeuralODE.py --num_epochs 1000 --noise_level 3e-1  --batch_time 4  --logging_root './Results' >> ./Results/Cubic2D/Cubic2D_NeuralODE_bt4.txt && echo "Done with NeuralODE with noise 3e-1"

python Cubic2D_NeuralODE.py --num_epochs 1000 --noise_level 4e-1  --batch_time 4  --logging_root './Results' >> ./Results/Cubic2D/Cubic2D_NeuralODE_bt4.txt && echo "Done with NeuralODE with noise 4e-1"

python Cubic2D_NeuralODE.py --num_epochs 1000 --noise_level 5e-1  --batch_time 4  --logging_root './Results' >> ./Results/Cubic2D/Cubic2D_NeuralODE_bt4.txt && echo "Done with NeuralODE with noise 5e-1"
