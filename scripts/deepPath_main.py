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

from env import Env # this is the original DeepPath Env class
from utils import *
## import necessary tensorflow classes

from tensorforce import TensorForceError
from tensorforce.agents import agents
from tensorforce.core.networks import from_json
from tensorforce.config import Configuration
from tensorforce.execution import Runner
from tensorforce.agents import agents
from Tforcedp import DPEnv

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--relation', help="Number of episodes")
    parser.add_argument('-a', '--agent', default='VPGAgent')
    parser.add_argument('-c', '--agent-config', default = './args/deepPath_agent_config.json', help="Agent configuration file")
    parser.add_argument('-n', '--network-config', default = './args/deepPath_agent_network.json', help="Network configuration file")
    parser.add_argument('-e', '--episodes', type=int, default=500, help="Number of episodes")
    parser.add_argument('-D', '--debug', action='store_true', default=False, help="Show debug outputs")
    parser.add_argument('-s', '--save', default = './DPAgents', help="Save agent to this dir (default ./DPAgents)")
    parser.add_argument('-se', '--save-episodes', type=int, default=100, help="Save agent every x episodes")


    """"
    Add support for following arguments after training is successful

    #parser.add_argument('-l', '--load', help="Load agent from this dir")
    """

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG) #can be changed

    ## Parse for following Arguments
    ##
    ## relation
    ## agent
    ## agent Configuration JSON file
    ## network configuration JSON file
    ## number of episodes - default to 500
    ##

    ## if relation name is not provided  as an input Parameter
    ## throw an Exception
    ## rest of the input parameters have the default value

    if args.relation: # relation is defined
       relation = args.relation
    else:
        logger.error("Error : No Relation name provided!")
        return

    if args.network_config:
        ## define network
       network = Configuration_from_json(args.network_config)
    else:
        logger.error("Error : No Network Config JSON file specified!")
        return


    if args.agent_config:
       agent_config = Configuration_from_json(args.agent_config)
       ## define agent using agent_config
       ## takes a dictionary containing variables for
       ## states, actions and network specs.
        agent = Agent.from_spec(
                        spec=agent_config,
                        kwargs=dict(
                            states_spec=environment.states, # TODO
                            actions_spec=environment.actions, # TODO
                            network_spec=network_spec # TODO
                        )
    )
    else:
        logger.error("Error : No Agent Config JSON file specified!")
        return


    graphpath = dataPath + 'tasks/' + relation + '/' + 'graph.txt'
    relationPath = dataPath + 'tasks/' + relation + '/' + 'train_pos'

    if args.debug:
        logger.info('-' * 16)
        logginer.info("Parameter")
        logger.info(relation)
        logger.info(agent_config)
        logger.info(network_config)
        logger.info(graphPath)
        logger.info(relationPath)


    if args.save:
    save_dir = os.path.dirname(args.save)
    if not os.path.isdir(save_dir):
        try:
            os.mkdir(save_dir, 0o755)
        except OSError:
            raise OSError("Error Saving Agent info in dir {} ()".format(save_dir))

    ## initialize the DeePath Environment class
    environment = DPEnv(graphPath, relationPath)

    ## define runner
    ## uncomment once the agent,network and env details are hashed out
    """
    runner = Runner(agent=agent,
                    environment = environment,
                    repeat_actions = 1,
                    save_path = args.save,
                    save_episodes = args.save_episodes)

    report_episodes = args.episodes/50 // default episodes = 500
    if args.debug
        report_episodes = 1

    def episode_finished(r):
        if r.episode % report_episodes == 0:
            sps = r.total_timesteps / (time.time() - r.start_time)
            logger.info("Finished episode {ep} after {ts} timesteps. Steps Per Second {sps}".format(ep=r.episode, ts=r.timestep, sps=sps))
            logger.info("Episode reward: {}".format(r.episode_rewards[-1]))
            logger.info("Average of last 50 rewards: {}".format(sum(r.episode_rewards[-50:]) / 50))
            logger.info("Average of last 100 rewards: {}".format(sum(r.episode_rewards[-100:]) / 100))
        return True


    logger.info("Starting {agent} for Environment '{env}'".format(agent=agent, env=environment))
    runner.run(args.episodes, args.max_timesteps, episode_finished=episode_finished)
    logger.info("Learning finished. Total episodes: {ep}".format(ep=runner.episode))

    """

    environment.close()


if __name__ == '__main__'
    main()
