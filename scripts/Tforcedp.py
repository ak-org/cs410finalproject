from utils import *
import numpy as np
import random
import datetime
from tensorforce.environments.environment import Environment
# ############################################################
# This file contains wrapper deeppath environment class
# mapped to Tensorforce's Environment class
# It will be called back in the deepPath_main.py file
# ############################################################

class DPEnv(Environment):
    def __init__(self, relationPath, graphPath, task=None):
        print ("calling init")
        self.task = task
        self.graphPath = graphPath
        self.relationPath = relationPath
        f1 = open(dataPath + 'entity2id.txt')
        f2 = open(dataPath + 'relation2id.txt')
        self.entity2id = f1.readlines()
        self.relation2id = f2.readlines()
        f1.close()
        f2.close()
        self.entity2id_ = {}
        self.relation2id_ = {}
        self.relations = []
        for line in self.entity2id:
            self.entity2id_[line.split()[0]] = int(line.split()[1])
        for line in self.relation2id:
            self.relation2id_[line.split()[0]] = int(line.split()[1])
            self.relations.append(line.split()[0])
        self.entity2vec = np.loadtxt(dataPath + 'entity2vec.bern')
        self.relation2vec = np.loadtxt(dataPath + 'relation2vec.bern')

        self.path = []
        self.path_relations = []

        self.action = dict(num_actions=action_space, type='int')

        # Knowledge Graph for path finding
        f = open(dataPath + 'kb_env_rl.txt')
        kb_all = f.readlines()
        f.close()

        self.kb = []
        for i in range(len(task)):
            if task[i] != None:
                relation = task[i].split()[2]
                for line in kb_all:
                    rel = line.split()[2]
                    if rel != relation and rel != relation + '_inv':
                        self.kb.append(line)

        print("processed KB")
        self.reset()



    def __str__(self):
        return 'DeepPath Env({}:{})'.format(self.relationPath, self.graphPath)

    def close(self):
        self.env = None


    def reset(self):
        print ("calling reset", datetime.datetime.now())
        sample = self.task[random.randint(0, len(self.task)-1)].split()
        self.localstate = [0, 0]
        self.localstate[0] = self.entity2id_[sample[0]]
        self.localstate[1] = self.entity2id_[sample[1]]
        self.state = self.localstate

        return self.state

    def execute(self, actions):
        '''
        This function process the interact from the agent
        state: is [current_position, target_position]
        action: an integer
        return: ([new_postion, target_position], done. reward)
        '''
        state = self.state
        print (self.state)

        done = 0  # Whether the episode has finished
        curr_pos = state[0]
        target_pos = state[1]
        chosen_relation = self.relations[actions]
        choices = []
        for line in self.kb:
            triple = line.rsplit()
            e1_idx = self.entity2id_[triple[0]]

            if curr_pos == e1_idx and triple[2] == chosen_relation and triple[1] in self.entity2id_:
                choices.append(triple)

        if len(choices) == 0:
            reward = -1
            next_state = state  # stay in the initial state
            done =1
            return (next_state, done,reward)
        else:  # find a valid step
            path = random.choice(choices)
            self.path.append(path[2] + ' -> ' + path[1])
            self.path_relations.append(path[2])
            # print 'Find a valid step', path
            # print 'Action index', action
            new_pos = self.entity2id_[path[1]]
            new_state = [new_pos, target_pos]
            self.state = new_state

            if new_pos == target_pos:
                print 'Find a path:', self.path
                done = 1
                reward = 1
                print ("Yay")
                new_state = None
            return (new_state, done, reward)


    def states(self, idx_list=None):
        return dict(shape=state_dim, type='float')



    def actions(self, entityID=0):
        return dict(num_actions=action_space, type='int')
