import numpy as np 

from tensorforce.env import Environment
from env import Env

'''
This file represents  knowledge graph environment mapped to Tensorforce's Environment class
It will be called back in the main code
'''
class kgenv(Environment):

    def __init__(self, datapath, task=None):
        '''
        Initialize Knowledge graph Env with datapth and tasks passed from 
        the parent class
        '''
        self.env = Env(datapath, task)
        

    def __str__(self):
        return 'DeepPath Env({}:{})'.format(self.datapath,self.task)
    
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
        reward, new_state, done = Env.interact(self, state, action)
        return reward, new_state, done

    def states(self):
        '''
        define states of the env here
        '''

    def actions(self):
        ''' 
        define actions of the env here
        '''

