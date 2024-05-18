#!/bin/bash

docker run -it --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /path/to/project/root/directory:/code lipschitz_rl:latest \
    /bin/sh -c "
    pwd; 
    cd /code/; 
    pip install -e .; 
    python ./scripts/pendulum/train_2_train_models.py;
    python ./scripts/pendulum/train_3_get_perturbed_rewards.py;
    "