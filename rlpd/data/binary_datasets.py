import os

import gym

try:
    import mj_envs
except:
    pass
import numpy as np

from rlpd.data.dataset import Dataset

AWAC_DATA_DIR = "~/.datasets/awac-data"


def process_expert_dataset(expert_datset):
    """This is a mess, but works"""
    all_observations = []
    all_next_observations = []
    all_actions = []
    all_rewards = []
    all_terminals = []

    for x in expert_datset:
        all_observations.append(
            np.vstack([xx["state_observation"] for xx in x["observations"]])
        )
        all_next_observations.append(
            np.vstack([xx["state_observation"] for xx in x["next_observations"]])
        )
        all_actions.append(np.vstack([xx for xx in x["actions"]]))
        # for some reason rewards has an extra entry, so in rlkit they just remove the last entry: https://github.com/rail-berkeley/rlkit/blob/354f14c707cc4eb7ed876215dd6235c6b30a2e2b/rlkit/demos/source/dict_to_mdp_path_loader.py#L84
        all_rewards.append(x["rewards"][:-1])
        all_terminals.append(x["terminals"])

    return {
        "observations": np.concatenate(all_observations, dtype=np.float32),
        "next_observations": np.concatenate(all_next_observations, dtype=np.float32),
        "actions": np.concatenate(all_actions, dtype=np.float32),
        "rewards": np.concatenate(all_rewards, dtype=np.float32),
        "terminals": np.concatenate(all_terminals, dtype=np.float32),
    }


def process_bc_dataset(bc_dataset):
    final_bc_dataset = {k: [] for k in bc_dataset[0] if "info" not in k}

    for x in bc_dataset:
        for k in final_bc_dataset:
            final_bc_dataset[k].append(x[k])

    return {
        k: np.concatenate(v, dtype=np.float32).squeeze()
        for k, v in final_bc_dataset.items()
    }


class BinaryDataset(Dataset):
    def __init__(
        self,
        env: gym.Env,
        clip_to_eps: bool = True,
        eps: float = 1e-5,
        remove_terminals=True,
        include_bc_data=True,
    ):

        env_prefix = env.spec.name.split("-")[0]

        expert_dataset = np.load(
            os.path.join(
                os.path.expanduser(AWAC_DATA_DIR), f"{env_prefix}2_sparse.npy"
            ),
            allow_pickle=True,
        )

        # this seems super random, but I grabbed it from here: https://github.com/rail-berkeley/rlkit/blob/c81509d982b4d52a6239e7bfe7d2540e3d3cd986/rlkit/launchers/experiments/awac/awac_rl.py#L124 and here https://github.com/rail-berkeley/rlkit/blob/354f14c707cc4eb7ed876215dd6235c6b30a2e2b/rlkit/demos/source/dict_to_mdp_path_loader.py#L153
        dataset_split = 0.9
        last_train_idx = int(dataset_split * len(expert_dataset))

        dataset_dict = process_expert_dataset(expert_dataset[:last_train_idx])

        if include_bc_data:
            bc_dataset = np.load(
                os.path.join(
                    os.path.expanduser(AWAC_DATA_DIR), f"{env_prefix}_bc_sparse4.npy"
                ),
                allow_pickle=True,
            )

            # this seems super random, but I grabbed it from here: https://github.com/rail-berkeley/rlkit/blob/c81509d982b4d52a6239e7bfe7d2540e3d3cd986/rlkit/launchers/experiments/awac/awac_rl.py#L124 and here https://github.com/rail-berkeley/rlkit/blob/354f14c707cc4eb7ed876215dd6235c6b30a2e2b/rlkit/demos/source/dict_to_mdp_path_loader.py#L153
            bc_dataset_split = 0.9
            bc_dataset = bc_dataset[: int(bc_dataset_split * len(bc_dataset))]
            bc_dataset = process_bc_dataset(bc_dataset)

            dataset_dict = {
                k: np.concatenate([dataset_dict[k], bc_dataset[k]])
                for k in dataset_dict
            }

        if clip_to_eps:
            lim = 1 - eps
            dataset_dict["actions"] = np.clip(dataset_dict["actions"], -lim, lim)

        dones = np.full_like(dataset_dict["rewards"], False, dtype=bool)

        for i in range(len(dones) - 1):
            if (
                np.linalg.norm(
                    dataset_dict["observations"][i + 1]
                    - dataset_dict["next_observations"][i]
                )
                > 1e-6
                or dataset_dict["terminals"][i] == 1.0
            ):
                dones[i] = True

        if remove_terminals:
            dataset_dict["terminals"] = np.zeros_like(dataset_dict["terminals"])

        dones[-1] = True

        dataset_dict["masks"] = 1.0 - dataset_dict["terminals"]
        del dataset_dict["terminals"]

        for k, v in dataset_dict.items():
            dataset_dict[k] = v.astype(np.float32)

        dataset_dict["dones"] = dones

        super().__init__(dataset_dict)
