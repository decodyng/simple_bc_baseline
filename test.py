
import json
import select
import time
import logging
import os
import threading


from typing import Callable

import aicrowd_helper
import gym
import minerl
import abc
import numpy as np
import torch as th
import coloredlogs
coloredlogs.install(logging.DEBUG)
from basalt_baselines.bc import bc_baseline, WRAPPERS as bc_wrappers
import realistic_benchmarks # TODO remove when Caves is ported into MineRL

# All the evaluations will be evaluated on MineRLObtainDiamondVectorObf-v0 environment
MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'FindCaves-v0')
MINERL_MAX_EVALUATION_EPISODES = int(os.getenv('MINERL_MAX_EVALUATION_EPISODES', 5))

# Parallel testing/inference, **you can override** below value based on compute
# requirements, etc to save OOM in this phase.
EVALUATION_THREAD_COUNT = int(os.getenv('EPISODES_EVALUATION_THREAD_COUNT', 2))
TRAINING_EXPERIMENT = bc_baseline

class EpisodeDone(Exception):
    pass

class Episode(gym.Env):
    """A class for a single episode.
    """
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self._done = False

    def reset(self):
        if not self._done:
            return self.env.reset()

    def step(self, action):
        s,r,d,i = self.env.step(action)
        if d:
            self._done = True
            raise EpisodeDone()
        else:
            return s,r,d,i

    def wrap_env(self, wrappers):
        for wrapper in wrappers:
            self.env = wrapper(self.env)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space




# DO NOT CHANGE THIS CLASS, THIS IS THE BASE CLASS FOR YOUR AGENT.
class MineRLAgentBase(abc.ABC):
    """
    To compete in the competition, you are required to implement a
    SUBCLASS to this class.
    
    YOUR SUBMISSION WILL FAIL IF:
        * Rename this class
        * You do not implement a subclass to this class 

    This class enables the evaluator to run your agent in parallel, 
    so you should load your model only once in the 'load_agent' method.
    """

    @abc.abstractmethod
    def load_agent(self):
        """
        This method is called at the beginning of the evaluation.
        You should load your model and do any preprocessing here.
        THIS METHOD IS ONLY CALLED ONCE AT THE BEGINNING OF THE EVALUATION.
        DO NOT LOAD YOUR MODEL ANYWHERE ELSE.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def run_agent_on_episode(self, single_episode_env : Episode):
        """This method runs your agent on a SINGLE episode.

        You should just implement the standard environment interaction loop here:
            obs  = env.reset()
            while not done:
                env.step(self.agent.act(obs)) 
                ...
        
        NOTE: This method will be called in PARALLEL during evaluation.
            So, only store state in LOCAL variables.
            For example, if using an LSTM, don't store the hidden state in the class
            but as a local variable to the method.

        Args:
            env (gym.Env): The env your agent should interact with.
        """
        raise NotImplementedError()


#######################
# YOUR CODE GOES HERE #
#######################

class MineRLBehavioralCloningAgent(MineRLAgentBase):
    def load_agent(self):
        self.policy = th.load("train/trained_policy.pt")
        self.policy.eval()

    def run_agent_on_episode(self, single_episode_env : Episode):
        # Get wrappers used in BC somehow, and wrap environment with those
        single_episode_env.wrap_env(bc_wrappers)
        obs = single_episode_env.reset()
        done = False
        while not done:
            action, _, _ = self.policy.forward(th.from_numpy(obs).unsqueeze(0))
            try:
                obs, reward, done, _ = single_episode_env.step(th.squeeze(action))
            except EpisodeDone:
                done = True
                continue


class MineRLMatrixAgent(MineRLAgentBase):
    """
    An example random agent. 
    Note, you MUST subclass MineRLAgentBase.
    """

    def load_agent(self):
        """In this example we make a random matrix which
        we will use to multiply the state by to produce an action!

        This is where you could load a neural network.
        """
        # Some helpful constants from the environment.
        flat_video_obs_size = 64*64*3
        obs_size = 64
        ac_size = 64
        self.matrix = np.random.random(size=(ac_size, flat_video_obs_size + obs_size))*2 -1
        self.flatten_obs = lambda obs: np.concatenate([obs['pov'].flatten()/255.0, obs['vector'].flatten()])
        self.act = lambda flat_obs: {'vector': np.clip(self.matrix.dot(flat_obs), -1,1)}


    def run_agent_on_episode(self, single_episode_env : Episode):
        """Runs the agent on a SINGLE episode.

        Args:
            single_episode_env (Episode): The episode on which to run the agent.
        """
        obs = single_episode_env.reset()
        done = False
        while not done:
            obs,reward,done,_ = single_episode_env.step(self.act(self.flatten_obs(obs)))


class MineRLRandomAgent(MineRLAgentBase):
    """A random agent"""
    def load_agent(self):
        pass # Nothing to do, this agent is a random agent.

    def run_agent_on_episode(self, single_episode_env : Episode):
        obs = single_episode_env.reset()
        done = False
        while not done:
            random_act = single_episode_env.action_space.sample()
            single_episode_env.step(random_act)
        
#####################################################################
# IMPORTANT: SET THIS VARIABLE WITH THE AGENT CLASS YOU ARE USING   # 
######################################################################
AGENT_TO_TEST = MineRLBehavioralCloningAgent # MineRLMatrixAgent, MineRLRandomAgent, YourAgentHere



####################
# EVALUATION CODE  #
####################
def main():
    agent = AGENT_TO_TEST()
    assert isinstance(agent, MineRLAgentBase)
    agent.load_agent()

    assert MINERL_MAX_EVALUATION_EPISODES > 0
    assert EVALUATION_THREAD_COUNT > 0

    # Create the parallel envs (sequentially to prevent issues!)
    envs = [gym.make(MINERL_GYM_ENV) for _ in range(EVALUATION_THREAD_COUNT)]
    episodes_per_thread = [MINERL_MAX_EVALUATION_EPISODES // EVALUATION_THREAD_COUNT for _ in range(EVALUATION_THREAD_COUNT)]
    episodes_per_thread[-1] += MINERL_MAX_EVALUATION_EPISODES - EVALUATION_THREAD_COUNT *(MINERL_MAX_EVALUATION_EPISODES // EVALUATION_THREAD_COUNT)
    # A simple function to evaluate on episodes!

    agent.run_agent_on_episode(Episode(envs[0]))
    # def evaluate(i, env):
    #     print("[{}] Starting evaluator.".format(i))
    #     for i in range(episodes_per_thread[i]):
    #         try:
    #             agent.run_agent_on_episode(Episode(env))
    #         except EpisodeDone:
    #             print("[{}] Episode complete".format(i))
    #             pass
    
    # evaluator_threads = [threading.Thread(target=evaluate, args=(i, envs[i])) for i in range(EVALUATION_THREAD_COUNT)]
    # for thread in evaluator_threads:
    #     thread.start()
    #
    # # wait for the evaluation to finish
    # for thread in evaluator_threads:
    #     thread.join()

if __name__ == "__main__":
    main()
    

