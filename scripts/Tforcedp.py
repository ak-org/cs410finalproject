import numpy as np

from tensorforce.env import Environment
from env import Env
from utils import *

'''
This file contains wrapper deeppath environment class
mapped to Tensorforce's Environment class
It will be called back in the deepPath_main.py file

'''
class DPEnv(Environment):

    def __init__(self, relationPath, graphPath, task=None):
        '''
        Initialize Knowledge graph Env with datapth and tasks passed from
        the parent class
        '''

        f = open(relationPath)
        train_data = f.readlines()
        f.close()

        num_samples = len(train_data)
        self.env = Env(dataPath, train_data[episode%num)saples])
        print ' '


    def __str__(self):
        return 'DeepPath Env({}:{})'.format(self.relationPath, self.graphPath)

    def close(self):
        self.env = None

    def reset(self):
        self.entity2id = {}
        self.relation2id = {}
        self.relations = {}
        self.entity2id_ = {}
        self.relation2id_ = {}
        self.relations_ = {}
        self.entity2vec = np.empty()
        self.relation2vec = np.empty()
        self.die = 0 #same as in init state

    def execute(self, state, action):
        
        return reward, new_state, done

    def states(self):
        '''
        define states of the env here
        '''

    def actions(self):
        '''
        define actions of the env here
        '''
