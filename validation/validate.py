import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime
import sys
import argparse
import warnings

torch.manual_seed(0)
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.sans-serif": "Latin Modern Roman",
        "font.size": 14,
        "legend.fontsize": 14,
        "legend.title_fontsize": 14,
    }
)

sys.path.append("../")
from params import (
    init_ma_controller,
    init_mad_controller,
    init_ddpg_controller,
)

warnings.simplefilter("ignore", UserWarning)

parser = argparse.ArgumentParser()

parser.add_argument(
    "-c",
    "--controller_type",
    choices=["MAD", "MA", "DDPG"],
    required=True,
)
parser.add_argument("-t", "--tag", type=int, required=True)
parser.add_argument(
    "-v", "--validation_type", choices=["Training", "Generalization"], required=True
)

args = parser.parse_args()

CONTROLLER_TYPE = args.controller_type
TAG = args.tag
VALIDATION_TYPE = args.validation_type

warnings.simplefilter("ignore", UserWarning)


if __name__ == "__main__":

    # Initialize Agent
    if CONTROLLER_TYPE == "MAD":
        DdpgAgent = init_mad_controller()
    elif CONTROLLER_TYPE == "MA":
        DdpgAgent = init_ma_controller()
    elif CONTROLLER_TYPE == "DDPG":
        DdpgAgent = init_ddpg_controller()

    # Defining Rewards Metric
    num_trajectories = 10
    len_trajectories = 500
    agent_1_initial_low = torch.FloatTensor([-3.0, -3.0, 0.0, 0.0])
    agent_2_initial_low = torch.FloatTensor([1.0, -3.0, 0.0, 0.0])
    agent_1_initial_high = torch.FloatTensor([-1.0, -1.0, 0.0, 0.0])
    agent_2_initial_high = torch.FloatTensor([3.0, -1.0, 0.0, 0.0])

    models_folder = f"../ddpg_models/{CONTROLLER_TYPE}_{TAG}"
    file_list = sorted(os.listdir(models_folder))

    if VALIDATION_TYPE == "Training":
        initial_state_low = torch.cat([agent_1_initial_low, agent_2_initial_low])
        initial_state_high = torch.cat([agent_1_initial_high, agent_2_initial_high])
        print(f"Started Training Validation for {models_folder}")

    if VALIDATION_TYPE == "Generalization":
        initial_state_low = torch.cat([agent_2_initial_low, agent_1_initial_low])
        initial_state_high = torch.cat([agent_2_initial_high, agent_1_initial_high])
        print(f"Started Generalization Validation for {models_folder}")

    start = datetime.datetime.now()
    datetime_format = "%Y-%m-%d %H:%M:%S"
    print(f"Start Time: {start.strftime(datetime_format)}")

    df_list = []
    for file in file_list:

        if file.split(".")[-1] != "pth":
            continue

        filename = os.path.join(models_folder, file)
        DdpgAgent.load_model_weight(filename=filename)

        row_list = [file]
        cumulative_rewards = np.zeros(num_trajectories)
        for i in range(num_trajectories):
            initial_state = (
                torch.rand(DdpgAgent.num_states)
                * (initial_state_high - initial_state_low)
                + initial_state_low
            )
            rewards_list, obs_list, action_list, w_list = DdpgAgent.get_trajectory(
                initial_state=initial_state, timesteps=len_trajectories
            )
            cumulative_rewards[i] = np.sum(np.array(rewards_list))
            row_list.append(cumulative_rewards[i])

        model_reward_mean = np.mean(cumulative_rewards)
        model_reward_std = np.std(cumulative_rewards)

        row_list.append(model_reward_mean)
        row_list.append(model_reward_std)
        df_list.append(row_list)

    column_list = ["model_name"]
    for i in range(num_trajectories):
        column_list.append(f"trajectory_{i+1}")
    column_list.append("model_reward_mean")
    column_list.append("model_reward_std")

    df = pd.DataFrame(df_list, columns=column_list)

    if VALIDATION_TYPE == "Training":
        save_path = os.path.join(models_folder, "training_rewards.csv")
        df.to_csv(save_path, index=False)
        print(f"Saved Training Rewards CSV at {save_path}.")

    if VALIDATION_TYPE == "Generalization":
        save_path = os.path.join(models_folder, "generalization_rewards.csv")
        df.to_csv(save_path, index=False)
        print(f"Saved Generalization Rewards CSV at {save_path}.")

    end = datetime.datetime.now()
    print(f"End Time: {end.strftime(datetime_format)}")

    print(
        f"Total Validation Duration: {divmod((end-start).total_seconds(), 60)[0]} minutes."
    )
