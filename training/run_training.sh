#!/bin/bash

# Define the grid of argument values
controller_type_values=("MAD" "AD" "MA" "DDPG")
tag=1
episodes=4

# Loop through controller types
for controller in "${controller_type_values[@]}"; do
        echo "Running train.py with controller_type=$controller, tag=$tag, episodes=$episodes"
        python3 train.py -c "$controller" -t "$tag" -e "$episodes"
done
