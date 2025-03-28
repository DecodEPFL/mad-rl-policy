#!/bin/bash

# Define the grid of argument values
controller_type_values=("MAD" "MA" "DDPG")
tag=1
episodes=1000

# Loop through controller types
for controller in "${controller_type_values[@]}"; do
        echo "Running train_continue.py with controller_type=$controller, tag=$tag, episodes=$episodes"
        python3 train_continue.py -c "$controller" -t "$tag" -e "$episodes"
done
