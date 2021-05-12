import gym
import numpy as np
import minerl
from copy import deepcopy


class DummyEnv(gym.Env):
    """
    A simplistic class that lets us mock up a gym Environment that is sufficient for our purposes
    without actually going through the whole convoluted registration process.
    """
    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space

    def step(self, action):
        if isinstance(self.action_space, gym.spaces.Dict):
            assert isinstance(action, dict)
        return self.observation_space.sample(), 0, True, {}

    def reset(self):
        return self.observation_space.sample()


class ExtractPOVAndTranspose(gym.ObservationWrapper):
    """
    Basically what it says on the tin. Extracts only the POV observation out of the `obs` dict,
    and transposes those observations to be in the (C, H, W) format used by stable_baselines and imitation
    """
    def __init__(self, env):
        super().__init__(env)
        non_transposed_shape = self.env.observation_space['pov'].shape
        self.high = np.max(self.env.observation_space['pov'].high)
        transposed_shape = (non_transposed_shape[2],
                            non_transposed_shape[0],
                            non_transposed_shape[1])
        # Note: this assumes the Box is of the form where low/high values are vector but need to be scalar
        transposed_obs_space = gym.spaces.Box(low=np.min(self.env.observation_space['pov'].low),
                                              high=np.max(self.env.observation_space['pov'].high),
                                              shape=transposed_shape,
                                              dtype=np.uint8)
        self.observation_space = transposed_obs_space

    def observation(self, obs):
        # Minecraft returns shapes in NHWC by default
        return np.swapaxes(obs['pov'], -1, -3)


class Testing10000StepLimitWrapper(gym.wrappers.TimeLimit):
    """
    A simple wrapper to impose a 10,000 step limit, for environments that don't have one built in
    """
    def __init__(self, env):
        super().__init__(env, 10000)


def wrap_env(env, wrappers):
    """
    Wrap `env` in all gym wrappers specified by `wrappers`
    """
    for wrapper in wrappers:
        env = wrapper(env)
    return env


def optional_observation_map(env, inner_obs):
    """
    If the env implements the `observation` function (i.e. if one of the
    wrappers is an ObservationWrapper), call that `observation` transformation
    on the observation produced by the inner environment
    """
    if hasattr(env, 'observation'):
        return env.observation(inner_obs)
    else:
        return inner_obs


def optional_action_map(env, inner_action):
    """
    This is doing something slightly tricky that is explained in the documentation for
    RecursiveActionWrapper (which TODO should eventually be in MineRL)
    Basically, it needs to apply `reverse_action` transformations from the inside out
    when converting the actions stored and used in a dataset

    """
    if hasattr(env, 'wrap_action'):
        return env.wrap_action(inner_action)
    else:
        return inner_action


def recursive_squeeze(dictlike):
    """
    Take a possibly-nested dictionary-like object of which all leaf elements are numpy ar
    """
    out = {}
    for k, v in dictlike.items():
        if isinstance(v, dict):
            out[k] = recursive_squeeze(v)
        else:
            out[k] = np.squeeze(v)
    return out


def get_data_pipeline_and_env(task_name, data_root, wrappers):
    """
    This code loads a data pipeline object and creates a dummy environment with the
    same observation and action space as the (wrapped) environment you want to train on

    :param task_name: The name of the MineRL task you want to get data for
    :param data_root: For manually specifying a MineRL data root
    :param wrappers: The wrappers you want to apply to both the loaded data and live environment
    """
    data_pipeline = minerl.data.make(environment=task_name,
                                     data_dir=data_root)
    dummy_env = DummyEnv(action_space=data_pipeline.action_space,
                         observation_space=data_pipeline.observation_space)
    wrapped_dummy_env = wrap_env(dummy_env, wrappers)
    return data_pipeline, wrapped_dummy_env


def create_data_iterator(wrapped_dummy_env, data_pipeline, batch_size, num_epochs, n_traj, remove_no_ops=False):
    """
    Construct a data iterator that (1) loads data from disk, and (2) wraps it in the set of
    wrappers that have been applied to `wrapped_dummy_env`.

    :param wrapped_dummy_env: An environment that mimics the base environment and wrappers we'll be using for training,
    but doesn't actually call Minecraft
    :param data_pipeline: A MineRL DataPipeline object that can handle loading data from disk
    :param batch_size: The batch size we want the iterator to produce
    :param num_epochs: The number of epochs we want the underlying iterator to run for
    :param n_traj: The number of trajectories we want to load; should be >= the batch size
    :param remove_no_ops: Whether to remove transitions with no-op demonstrator actions from batches as they are generated

    :yield: wrapped observations and actions
    """

    if num_epochs is None:
        num_epochs = 100
        print("Training with an undefined number of epochs (defined number of batches), using 100-epoch data iterator")

    assert n_traj >= batch_size, "You need to run with more trajectories than your batch size"
    for current_obs, action, reward, next_obs, done in data_pipeline.batch_iter(batch_size=batch_size,
                                                                                num_epochs=num_epochs,
                                                                                seq_len=1,
                                                                                epoch_size=n_traj):
        wrapped_obs = optional_observation_map(wrapped_dummy_env,
                                               recursive_squeeze(current_obs))
        wrapped_next_obs = optional_observation_map(wrapped_dummy_env,
                                                    recursive_squeeze(next_obs))
        wrapped_action = optional_action_map(wrapped_dummy_env,
                                             recursive_squeeze(action))

        if remove_no_ops:
            # This definitely makes assumptions about the action space, namely that all-zeros corresponds to a no-op
            not_no_op_indices = wrapped_action.sum(axis=1) != 0
            wrapped_obs = wrapped_obs[not_no_op_indices]
            wrapped_next_obs = wrapped_next_obs[not_no_op_indices]
            wrapped_action = wrapped_action[not_no_op_indices]

        yield dict(obs=wrapped_obs,
                   acts=wrapped_action,
                   rew=reward,
                   next_obs=wrapped_next_obs,
                   dones=done)
