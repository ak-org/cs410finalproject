from utils import *
import numpy as np
import random
from tensorforce.environments.environment import Environment

###############################################################################################################
##  In reinforcement learning, there is an agent - who interacts with
##  a given Environment. Environment accepts current and future step and
##  returns reward value back to the agent
##
##  Env object has the following instance variables
##          -- entity2id_ a dictionary populated from entity2id.txt file
##          -- relation2id_ a dictionary populated from relation2id.txt file
##          -- relations a vector populated from the relations in relation2id.txt file
##          -- entity2vec populated from entity2vec.bern file
##          -- relation2vec populated from relation2vec.bern file
##          -- kb knowledge graph for path finding. This stores all relations from
##                  kb_env_rl.txt expect one corresponding to input task and its inverse.
##                  Relation from state to A->B as R and the relation from B->A as R_inverse
##          -- die number of times the agent chose the wrong path
##
##  Env is defined as class containing following methods
##  -- Initialize init()
##          Reads the entity2id.txt and relation2id.txt files and populates the 2 dictionaries self.entity2id_
##          and self.relation2id_variables, stores the relations from relation2id.txt in self.relation,
##          populates entity2vec and relation2vec from corresponding .bern files. Populates the kb object with
##          relations (corresponding to input task) from kb_env_rl.txt. Sets variable die to 0.
##
##  -- execute()
##          Called during learning reinforcement phases
##          state: is [current_position, target_position]
##		    action: an integer
##          return: (reward, [new_postion, target_position], done)
##
##  -- states()
##          takes as input current or next_state
##          returns the current and (target-current) positions as an array
##
##  -- actions()
##          takes the entity id as input
##          Returns set of valid actions for a given state from self.kb
##          in form of an array
##
##  -- reset()
##          Called from Runner before begining training
##
##  -- close()
##          Set environment to None

##
##
##
############################################################################################################

class DPEnv(Environment):
    def __init__(self, relationPath, graphPath, task=None):
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
        sample = task[0].split()

        localstate=[0,0]
        localstate[0] = self.entity2id_[sample[0]]
        localstate[1] = self.entity2id_[sample[1]]
        self.state = localstate
        self.action =dict(num_actions=action_space, type='int')


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
        self.die = 0  # record how many times does the agent choose an invalid path


    def __str__(self):
        return 'DeepPath Env({}:{})'.format(self.relationPath, self.graphPath)

    def close(self):
        self.env = None

    def reset(self):
        state = np.zeros(state_dim)
        return state

    def execute(self, actions):
        '''
        This function process the interact from the agent
        state: is [current_position, target_position]
        action: an integer
        return: ([new_postion, target_position], done. reward)
        '''
        state = self.state

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
            self.die += 1
            next_state = state  # stay in the initial state
            next_state[-1] = self.die
            return (next_state, done,reward)
        else:  # find a valid step
            path = random.choice(choices)
            self.path.append(path[2] + ' -> ' + path[1])
            self.path_relations.append(path[2])
            # print 'Find a valid step', path
            # print 'Action index', action
            self.die = 0
            new_pos = self.entity2id_[path[1]]
            reward = 0
            #new_state = [new_pos, target_pos, self.die]
            new_state = [new_pos, target_pos]

            if new_pos == target_pos:
                print 'Find a path:', self.path
                done = 1
                reward = 0
                new_state = None
            return (new_state, done, reward)


    def states(self, idx_list=None):
        localstate = [0,0]
        if idx_list != None:
            curr = self.entity2vec[idx_list[0], :]
            targ = self.entity2vec[idx_list[1], :]
            localstate[0] = curr
            localstate[1] = targ - curr
            self.state = localstate
        return self.state


    def actions(self, entityID=0):
        localactions = set()
        for line in self.kb:
            triple = line.split()
            e1_idx = self.entity2id_[triple[0]]
            if e1_idx == entityID:
                localactions.add(self.relation2id_[triple[2]])
        self.action = localactions
        return np.array(list(localactions))
