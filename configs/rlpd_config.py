from configs import sac_config


def get_config():
    config = sac_config.get_config()

    config.num_qs = 10
    config.num_min_qs = 2
    config.critic_layer_norm=True

    return config
