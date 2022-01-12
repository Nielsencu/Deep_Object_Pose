sudo cp ../hyperparameters.json /opt/ml/input/config/
sudo cp ../hyperparameters_withoutgen.json /opt/ml/input/config/
torchrun train.py train

