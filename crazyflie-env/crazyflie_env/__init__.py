from gym.envs.registration import register

register(
    id = "CrazyflieEnv-v0",
    entry_point = "crazyflie_env.envs:CrazyflieEnv",
)