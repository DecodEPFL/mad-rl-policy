import torch
import argparse
import sys
import os
import time
import datetime
import warnings

sys.path.append("../")
from params import (
    init_ma_controller,
    init_ad_controller,
    init_mad_controller,
    init_ddpg_controller,
)

warnings.simplefilter("ignore", UserWarning)

parser = argparse.ArgumentParser()

parser.add_argument(
    "-c",
    "--controller_type",
    choices=["MAD", "AD", "MA", "DDPG"],
    required=True,
)
parser.add_argument("-t", "--tag", type=int, required=True)
parser.add_argument("-e", "--episodes", type=int, required=True)

args = parser.parse_args()

CONTROLLER_TYPE = args.controller_type
TAG = args.tag
EPISODES = args.episodes

# Defining Training
num_episodes = EPISODES
len_episode = 500
save_frequency = 10
cooldown_frequency = 100
cooldown_duration = 2.5 * 60
save_folder = f"../ddpg_models/{CONTROLLER_TYPE}_{TAG}"

# Initialize Agent
if CONTROLLER_TYPE == "MAD":
    DdpgAgent = init_mad_controller()
elif CONTROLLER_TYPE == "AD":
    DdpgAgent = init_ad_controller()
elif CONTROLLER_TYPE == "MA":
    DdpgAgent = init_ma_controller()
elif CONTROLLER_TYPE == "DDPG":
    DdpgAgent = init_ddpg_controller()

files_list_ordered = sorted(os.listdir(save_folder))
models_list_ordered = []
for f in files_list_ordered:
    if f.split(".")[-1] == "pth":
        models_list_ordered.append(f)
last_episode_model = models_list_ordered[-1]
print(f"Latest Trained Episode Model: {last_episode_model}")

latest_weights = os.path.join(save_folder, last_episode_model)
DdpgAgent.load_model_weight(filename=latest_weights)

last_ep = int(last_episode_model.split(".")[0].split("_")[-1])
DdpgAgent.episode_count = last_ep

print(
    f"Started Training for {num_episodes} Episodes: Saving every {save_frequency} episodes at {save_folder}."
)

start = datetime.datetime.now()
datetime_format = "%Y-%m-%d %H:%M:%S"
print(f"Start Time: {start.strftime(datetime_format)}")

for i in range(num_episodes):

    DdpgAgent.train(
        total_episodes=1,
        episode_length=len_episode,
    )

    if DdpgAgent.episode_count % save_frequency == 0:
        DdpgAgent.save_model_weights(
            filename=f"{save_folder}/ep_{DdpgAgent.episode_count:05d}.pth"
        )

    if DdpgAgent.episode_count % cooldown_frequency == 0:
        print(
            f"Cooldown Period of {cooldown_duration/60} minutes started at {datetime.datetime.now().strftime(datetime_format)}"
        )
        time.sleep(cooldown_duration)

end = datetime.datetime.now()
print(f"End Time: {end.strftime(datetime_format)}")

print(f"Total Training Duration: {divmod((end-start).total_seconds(), 60)[0]} minutes.")
