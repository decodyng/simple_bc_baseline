import minerl
from sacred import Experiment
import basalt_utils.wrappers as wrapper_utils
from stable_baselines3.common.torch_layers import NatureCNN
from minerl.herobraine.wrappers.video_recording_wrapper import VideoRecordingWrapper
from basalt_utils.sb3_compat.policies import SpaceFlatteningActorCriticPolicy
from basalt_utils.sb3_compat.cnns import MAGICALCNN
from basalt_utils.callbacks import BatchEndIntermediateRolloutEvaluator
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
import collections
from imitation.algorithms.bc import BC
import imitation.data.rollout as il_rollout
import logging
import torch as th
from basalt_utils import utils
import os
import imitation.util.logger as imitation_logger
from sacred.observers import FileStorageObserver
from time import time

bc_baseline = Experiment("basalt_bc_baseline")

WRAPPERS = [# Transforms continuous camera action into discrete up/down/no-change buckets on both pitch and yaw
            (wrapper_utils.CameraDiscretizationWrapper, dict()),
            # Flattens a Dict action space into a Box, but retains memory of how to expand back out
            (wrapper_utils.ActionFlatteningWrapper, dict()),
            # Pull out only the POV observation from the observation space; transpose axes for SB3 compatibility
            (utils.ExtractPOVAndTranspose, dict())] #,

            # Add a time limit to the environment (only relevant for testing)
            # utils.Testing10000StepLimitWrapper,
            # wrapper_utils.FrameSkip]


@bc_baseline.config
def default_config():
    task_name = "MineRLFindCaves-v0"
    train_batches = None
    train_epochs = None
    log_interval = 1
    data_root = os.getenv('MINERL_DATA_ROOT', "/Users/cody/Code/il-representations/data/minecraft")
    # SpaceFlatteningActorCriticPolicy is a policy that supports a flattened Dict action space by
    # maintaining multiple sub-distributions and merging their results
    policy_class = SpaceFlatteningActorCriticPolicy
    wrappers = WRAPPERS
    save_location = "/Users/cody/Code/simple_bc_baseline/results"
    policy_path = 'trained_policy.pt'
    use_rollout_callback = False
    callback_batch_interval = 1000
    callback_rollouts = 5
    save_videos = False
    mode = 'train'
    test_policy_path = None
    test_n_rollouts = None
    batch_size = 32
    n_traj = None
    lr = 1e-4
    _ = locals()
    del _

@bc_baseline.named_config
def normal_policy_class():
    policy_class = ActorCriticCnnPolicy
    _ = locals()
    del _


@bc_baseline.main
def main(mode):
    if mode == 'train':
        train_bc()
    if mode == 'test':
        test_bc()


@bc_baseline.capture
def test_bc(task_name, data_root, wrappers, test_policy_path, test_n_rollouts):
    data_pipeline, wrapped_env = utils.get_data_pipeline_and_env(task_name, data_root, wrappers, dummy=False)
    vec_env = DummyVecEnv([lambda: wrapped_env])
    policy = th.load(test_policy_path)
    trajectories = il_rollout.generate_trajectories(policy, vec_env, il_rollout.min_episodes(test_n_rollouts))
    stats = il_rollout.rollout_stats(trajectories)
    stats = collections.OrderedDict([(key, stats[key])
                                     for key in sorted(stats)])

    # print it out
    kv_message = '\n'.join(f"  {key}={value}"
                           for key, value in stats.items())
    logging.info(f"Evaluation stats on '{task_name}': {kv_message}")


@bc_baseline.capture
def train_bc(task_name, batch_size, data_root, wrappers, train_epochs, n_traj, lr,
             policy_class, train_batches, log_interval, save_location, policy_path,
             use_rollout_callback, callback_batch_interval, callback_rollouts, save_videos):

    # This code is designed to let you either train for a fixed number of batches, or for a fixed number of epochs
    assert train_epochs is None or train_batches is None, \
        "Only one of train_batches or train_epochs should be set"
    assert not (train_batches is None and train_epochs is None), \
        "You cannot have both train_epochs and train_epochs set to None"

    run_save_location = os.path.join(save_location, str(round(time())))
    os.mkdir(run_save_location)

    # This `get_data_pipeline_and_env` utility is designed to be shared across multiple baselines
    # It takes in a task name, data root, and set of wrappers and returns
    # TODO clean up this documentation
    # (1) A "Dummy Env", i.e. an env with the same environment spaces as you'd getting from making the env associated
    #     with this task and wrapping it in `wrappers`, but without having to actually start up Minecraft
    # (2) A MineRL DataPipeline that can be used to construct a batch_iter used by BC, and also as a handle to clean
    #     up that iterator after training.
    if save_videos:
        wrappers = [(VideoRecordingWrapper, {'video_directory':
                                                 os.path.join(run_save_location, 'videos')})] + wrappers
    data_pipeline, wrapped_env = utils.get_data_pipeline_and_env(task_name, data_root, wrappers,
                                                                 dummy=not use_rollout_callback)

    # This utility creates a data iterator that is basically a light wrapper around the baseline MineRL data iterator
    # that additionally:
    # (1) Applies all observation and action transformations specified by the wrappers in `wrappers`, and
    # (2) Calls `np.squeeze` recursively on all the nested dict spaces to remove the sequence dimension, since we're
    #     just doing single-frame BC here
    data_iter = utils.create_data_iterator(wrapped_env, data_pipeline, batch_size, train_epochs, n_traj)
    if policy_class == SpaceFlatteningActorCriticPolicy:
        policy = policy_class(observation_space=wrapped_env.observation_space,
                              action_space=wrapped_env.action_space,
                              env=wrapped_env,
                              lr_schedule=lambda _: 1e-4,
                              features_extractor_class=MAGICALCNN)
    else:
        policy = policy_class(observation_space=wrapped_env.observation_space,
                              action_space=wrapped_env.action_space,
                              lr_schedule=lambda _: 1e-4,
                              features_extractor_class=MAGICALCNN)

    imitation_logger.configure(run_save_location, ["stdout", "tensorboard"])
    if use_rollout_callback:
        callback = BatchEndIntermediateRolloutEvaluator(policy=policy,
                                                        env=wrapped_env,
                                                        save_dir=os.path.join(run_save_location, 'policy'),
                                                        evaluate_interval_batches=callback_batch_interval,
                                                        n_rollouts=callback_rollouts)
    else:
        callback = None

    bc_trainer = BC(
        observation_space=wrapped_env.observation_space,
        action_space=wrapped_env.action_space,
        policy_class= lambda **kwargs: policy,
        policy_kwargs=None,
        expert_data=data_iter,
        device='auto',
        optimizer_cls=th.optim.Adam,
        optimizer_kwargs=dict(lr=lr),
        ent_weight=1e-3,
        l2_weight=1e-5)
    bc_trainer.train(n_epochs=train_epochs,
                     n_batches=train_batches,
                     log_interval=log_interval,
                     on_batch_end=callback)
    bc_trainer.save_policy(policy_path=os.path.join(run_save_location, policy_path))
    bc_baseline.add_artifact(os.path.join(run_save_location, policy_path))
    bc_baseline.log_scalar(f'run_location={run_save_location}', 1)
    print("Training complete; cleaning up data pipeline!")
    #data_pipeline.close()


if __name__ == "__main__":
    bc_baseline.observers.append(FileStorageObserver("sacred_results"))
    bc_baseline.run_commandline()