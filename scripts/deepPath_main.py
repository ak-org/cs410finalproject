"""
DeepPath main function
based on Tensorforce
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import sys
import time

import numpy as np
import tensorforce

from env import Env  # this is the original DeepPath Env class
from utils import *
## import necessary tensorflow classes

from tensorforce import TensorForceError
from tensorforce.agents import VPGAgent
from tensorforce.agents import DQNAgent
from tensorforce.execution import Runner
from tensorforce.agents import agents

from Tforcedp import DPEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--relation', help="Number of episodes")
    parser.add_argument('-e', '--episodes', type=int, default=500, help="Number of episodes")
    # parser.add_argument('-s', '--save', default = './DPAgents', help="Save agent to this dir (default ./DPAgents)")
    # parser.add_argument('-se', '--save-episodes', type=int, default=100, help="Save agent every x episodes")
    parser.add_argument('-D', '--debug', action='store_true', default=False, help="Show debug outputs")

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # can be changed

    if args.relation:  # relation is defined
        relation = args.relation
    else:
        logger.error("Error : No Relation name provided!")
        return

    graphPath = dataPath + 'tasks/' + relation + '/' + 'graph.txt'
    relationPath = dataPath + 'tasks/' + relation + '/' + 'train_pos'
    f = open(relationPath)
    data = f.readlines()
    f.close()

    num_samples = len(data)


    ## initialize the DeePath Environment class
    environment = DPEnv(graphPath, relationPath, task=data)

    network_spec = [
        dict(type='dense', size=512, activation='relu'),
        dict(type='dense', size=1024, activation='relu')
    ]

    step_optimizer = dict(type='adam', learning_rate=1e-3)


    agent = VPGAgent(states_spec=environment.state(),
                     actions_spec=environment.actions(),
                     network_spec=network_spec, optimizer=step_optimizer,
                     discount=0.99, batch_size=1000)
    # agent = DQNAgent(states_spec=dict(shape=(1, state_dim), type='float'),
    #                  actions_spec=dict(num_actions=action_space, type='int'),
    #                  network_spec=network_spec, optimizer=step_optimizer,
    #                  discount=0.99, batch_size=1000)

    runner = Runner(agent=agent, environment=environment)

    report_episodes = args.episodes / 500  # default episodes = 500

    def episode_finished(r):
        if r.episode % report_episodes == 0:
            sps = r.total_timesteps / (time.time() - r.start_time)
            print(
                "Finished episode {ep} after {ts} timesteps. Steps Per Second {sps}".format(ep=r.episode, ts=r.timestep,
                                                                                            sps=sps))
            print("Episode reward: {}".format(r.episode_rewards[-1]))
            print("Average of last 50 rewards: {}".format(sum(r.episode_rewards[-50:]) / 50))
            print("Average of last 100 rewards: {}".format(sum(r.episode_rewards[-100:]) / 100))
        return True

    print("Starting {agent} for Environment '{env}'".format(agent=agent, env=environment))
    runner.run(args.episodes, 1, episode_finished=episode_finished)
    print("Learning finished. Total episodes: {ep}".format(ep=runner.episode))
    environment.close()


if __name__ == '__main__':
    main()
