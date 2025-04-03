#!/bin/bash

# Define the grid of argument values
controller_type_values=("MAD" "AD" "MA" "DDPG")
tag=1
validation_type="Generalization_losses"        #("Training_losses" "Generalization_losses")

# Loop through controller types
for controller in "${controller_type_values[@]}"; do
        echo "Running validate_losses.py with controller_type=$controller, tag=$tag, validation_type=$validation_type"
        python3 validate_losses.py -c "$controller" -t "$tag" -v "$validation_type"
done
