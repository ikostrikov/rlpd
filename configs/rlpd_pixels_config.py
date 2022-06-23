from ml_collections.config_dict import config_dict

from configs import drq_config


def get_config():
    config = drq_config.get_config()

    config.num_qs = 10
    config.num_min_qs = 1

    config.critic_layer_norm = True
    config.backup_entropy = False

    return config
