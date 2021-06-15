import gym
from sacred import Experiment

import basalt_utils.wrappers as wrapper_utils
from basalt_utils import utils
from basalt_utils.sb3_compat.policies import SpaceFlatteningActorCriticPolicy
from drlhp import HumanPreferencesEnvWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import NatureCNN
import minerl

WRAPPERS = [# Transforms continuous camera action into discrete up/down/no-change buckets on both pitch and yaw
            (wrapper_utils.CameraDiscretizationWrapper, dict()),
            # Flattens a Dict action space into a Box, but retains memory of how to expand back out
            (wrapper_utils.ActionFlatteningWrapper, dict()),
            # Pull out only the POV observation from the observation space; transpose axes for SB3 compatibility
            (utils.ExtractPOVAndTranspose, dict()),
            (wrapper_utils.FrameSkip, dict(n_repeat=4))]

drlhp_baseline = Experiment("basalt_drlhp_baseline")

@drlhp_baseline.config
def default_config():
    task_name = "MineRLBasaltFindCave-v0"
    train_batches = 10
    train_epochs = None
    log_interval = 1
    data_root = "/Users/cody/Code/il-representations/data/minecraft"
    # SpaceFlatteningActorCriticPolicy is a policy that supports a flattened Dict action space by
    # maintaining multiple sub-distributions and merging their results
    policy_class = SpaceFlatteningActorCriticPolicy
    wrappers = WRAPPERS
    save_location = "/Users/cody/Code/simple_bc_baseline/results"
    policy_path = 'trained_policy.pt'
    batch_size = 16
    n_traj = 16
    lr = 1e-4
    _ = locals()
    del _


@drlhp_baseline.automain
def train_drlhp(task_name, batch_size, data_root, wrappers, train_epochs, n_traj, lr,
                policy_class, train_batches, log_interval, save_location, policy_path):
    env = gym.make(task_name)
    wrapped_env = utils.wrap_env(env, wrappers)
    wrapped_env = HumanPreferencesEnvWrapper(wrapped_env,
                                             segment_length=100,
                                             synthetic_prefs=False,
                                             n_initial_training_steps=10)
    space_flattening_policy = SpaceFlatteningActorCriticPolicy(observation_space=wrapped_env.observation_space,
                                                               action_space=wrapped_env.action_space,
                                                               lr_schedule=lambda _: 1e100,
                                                               env=wrapped_env,
                                                               features_extractor_class=NatureCNN)
    def sfp(*args, **kwargs):
        return space_flattening_policy
    ppo_trainer = PPO(policy=sfp,
                      env=wrapped_env)
    ppo_trainer.learn(500, reset_num_timesteps=False)

