sudo cp ../hyperparameters.json /opt/ml/input/config/
python -m torch.distributed.launch --nproc_per_node=1 generate_train.py train
