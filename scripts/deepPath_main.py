from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
from utils import *

from tensorforce.agents import VPGAgent
from tensorforce.agents import DQNAgent
from tensorforce.execution import Runner
from Tforcedp import DPEnv


###############################################################################################################
##  This is the main function of the tensorforce
##  implementation of the Deep Path program
##
##  The program takes following argument as parameters
##
##  -r  or --relation = relation name
##  -e  or --episodes = Number of episodes
##                      default : 500
##  -r  or --relation = relation name
##  -a  or --agent    = Agent Name
##                      default : vpg
##                      allowed values : vpg or dqn (lowercase)
##  -D  or --debug    = Show Debug Logs
##                      default : False
##
##
##
##
##
##

############################################################################################################

def main():

    # Parser for program arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--relation', help="relation name")
    parser.add_argument('-e', '--episodes', type=int, default=500, help="Number of episodes")
    parser.add_argument('-a', '--agent', type=str, default='vpg', help="VPG or DQN Agent")
    # parser.add_argument('-s', '--save', default = './DPAgents', help="Save agent to this dir (default ./DPAgents)")
    # parser.add_argument('-se', '--save-episodes', type=int, default=100, help="Save agent every x episodes")
    parser.add_argument('-D', '--debug', action='store_true', default=False, help="Show debug outputs")

    # Parse program arguments
    args = parser.parse_args()

    # Enabled logging
    logger = logging.getLogger(__name__)

    # TODO - Change using program argument
    logger.setLevel(logging.DEBUG)

    # Check if relation is defined
    if args.relation:
        relation = args.relation
    # Throw error if relation is not defined
    else:
        logger.error("Error : No Relation name provided!")
        return

    # Set the path of the appropriate knowledge graph and relation path
    # This can change based on the data folder
    graphPath = dataPath + 'tasks/' + relation + '/' + 'graph.txt'
    relationPath = dataPath + 'tasks/' + relation + '/' + 'train_pos'

    # Read the train_pos and populate data object
    f = open(relationPath)
    data = f.readlines()
    f.close()

    # Initialize the DeePath Environment class for tensorforce
    environment = DPEnv(graphPath, relationPath, task=data)

    # Define network layer using tensorforce API
    network_spec = [
        dict(type='dense', size=512, activation='relu'),
        dict(type='dense', size=1024, activation='relu')
    ]

    # define adam optimizer and learning rate
    step_optimizer = dict(type='adam', learning_rate=1e-3)
    agent = None


    # Create the agent as given in the program argument
    if args.agent == 'vpg':
        agent = VPGAgent(states_spec=dict(shape=state_dim, type='float'),
                         actions_spec=dict(num_actions=action_space, type='int'),
                         network_spec=network_spec, optimizer=step_optimizer,
                         discount=0.99, batch_size=1000)
    elif args.agent == 'dqn':
        agent = DQNAgent(states_spec=dict(shape=state_dim, type='float'),
                         actions_spec=dict(num_actions=action_space, type='int'),
                         network_spec=network_spec, optimizer=step_optimizer,
                         discount=0.99, batch_size=1000)

    # Initialize tensorforce runner by passing agent and environment
    runner = Runner(agent=agent, environment=environment)

    # Define report_episodes for logging
    report_episodes = args.episodes / 500  # default episodes = 500


    # Inner function for logging the episode
    def episode_finished(r):
        if r.episode % report_episodes == 0:
            #sps = r.total_timesteps / (time.time() - r.start_time)
            print(
                "Finished episode {ep} after {ts} timesteps. Steps Per Second ".format(ep=r.episode, ts=r.timestep))
            print("Episode reward: {}".format(r.episode_rewards[-1]))
            print("Average of last 50 rewards: {}".format(sum(r.episode_rewards[-50:]) / 50))
            print("Average of last 100 rewards: {}".format(sum(r.episode_rewards[-100:]) / 100))
        return True

    print("Starting {agent} for Environment '{env}'".format(agent=agent, env=environment))

    # Invoke tensorforce runner
    runner.run(episodes = args.episodes, max_episode_timesteps=report_episodes, episode_finished=episode_finished)
    print("Learning finished. Total episodes: {ep}".format(ep=runner.episode))
    environment.close()

# default entry main function
if __name__ == '__main__':
    main()
