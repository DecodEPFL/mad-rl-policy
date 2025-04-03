#!/bin/bash

# Define the grid of argument values
controller_type_values=("MAD" "AD" "MA" "DDPG")
tag=1
validation_type="Training"        #("Training" "Generalization")

# Loop through controller types
for controller in "${controller_type_values[@]}"; do
        echo "Running validate.py with controller_type=$controller, tag=$tag, validation_type=$validation_type"
        python3 validate.py -c "$controller" -t "$tag" -v "$validation_type"
done
