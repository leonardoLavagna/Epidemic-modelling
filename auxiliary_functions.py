## AUXILIARY FUNCTIONS FOR EPIDEMIC SIMULATION
## © Leonardo Lavagna 2022

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import EoN
import random
import numpy as np
import pandas as pd


# DEFINE SIR-SIRV CLASS

class SIR:
    def __init__(self, graph, beta, gamma, recovery=""):
        self.g = graph.copy()   
        self.N = graph.order()
        self.BETA = beta  
        self.GAMMA = gamma      
        self.keys=['t', 'S', 'I', 'R', 'SI', 'IR']
        self.data = { k: [] for k in self.keys }
        self.data_normalized = { k: [] for k in self.keys}
        self.seed=0
        
    def random_immunization(self, frac):
        '''
        apply immunization at random
        '''
        n=int(self.N * frac)
        immune = list(np.random.choice(self.g.nodes(), size=n, replace=False))
        for node in immune:
            self.g.remove_node(node)
            
    def prioritized_immunization(self, frac):
        '''
        apply immunization ordering for higher degree centrality
        '''
        n=int(self.N * frac)
        
        degree_dict=nx.closeness_centrality(self.g)
 
        degree=sorted( [(k, degree_dict[k]) for k in degree_dict], key=lambda u:u[1], reverse=True )
        immune=[d[0] for d in degree[:n]]
        for node in immune:
            self.g.remove_node(node)
            
        
    def run(self,  frac, priority="", recovery="", seed=[], num_steps=1, nodes={}, g_sample={}, thresh=1):
        if priority=="random":
            self.random_immunization(frac)
        elif priority=="degree":
            self.prioritized_immunization(frac)
            
        if not len(seed):
            seed = list(np.random.choice(self.g.nodes(), size=1, replace=False))
            
        # initialize sets of S/I/R nodes
        I = set(seed)
        S = set(self.g.nodes()).difference(I)
        R = set()
        t = 0
        
        #S->I transition
        SI = set(seed) 
        #I->R transition
        IR = set()     
        
        while True:
            # generator logic: yield current status every num_steps iterations
            if t % num_steps == 0:
                data=[t, len(S), len(I), len(R), len(SI), len(IR)]
                for i in range(len(self.keys)):
                    k=self.keys[i]
                    self.data[k].append(data[i])
            # stop when there are no infectious nodes left
            if not len(I):
                # normalize data
                self.data_normalized = {k: [v/self.N for v in self.data[k]] for k in self.data if k not in ["t"]}
                break

            SI = set()
            IR = set()
            
            # loop over neighbors of infectious nodes
            for i in set(I):
                # TRANSMISSION
                for j in S.intersection(self.g.neighbors(i)):
                    # Bernoulli sampling
                    if np.random.uniform() < self.BETA:
                        S.remove(j)
                        I.add(j)
                        SI.add(j)
                        
                # RECOVERY                
                if recovery=="uniform":
                    if np.random.uniform() < np.random.uniform():
                        I.remove(i)
                        R.add(i)
                        IR.add(i)
                elif recovery=="gaussian":
                    if np.random.uniform() < np.random.normal():
                        I.remove(i)
                        R.add(i)
                        IR.add(i)
                elif recovery=="power":
                    if np.random.uniform() < np.random.power(a):
                        I.remove(i)
                        R.add(i)
                        IR.add(i)
                else:
                    # Bernoulli sampling
                    if np.random.uniform() < self.GAMMA:
                        I.remove(i)
                        R.add(i)
                        IR.add(i)
                    
            t += 1
            
    
    def getData(self):
        return self.data

    
    def getDataNormalized(self):
        return self.data_normalized

# END SIR-SIRV CLASS

# DEFINE SIRE CLASS

class SIRE:
    def __init__(self, graph, time, beta, gamma, mu, epsilon, recovery=""):
        self.g = graph.copy()   
        self.N = graph.order()
        self.BETA = beta  
        self.GAMMA = gamma 
        self.MU = mu
        self.EPSILON = epsilon
        self.T = time
        self.keys=['t', 'S', 'I', 'R', 'E', 'SI', 'IR', 'RE', 'EI']
        self.data = { k: [] for k in self.keys }
        self.data_normalized = { k: [] for k in self.keys}
        self.seed=0
        
    def random_immunization(self, frac):
        '''
        apply immunization at random
        '''
        n=int(self.N * frac)
        immune = list(np.random.choice(self.g.nodes(), size=n, replace=False))
        for node in immune:
            self.g.remove_node(node)
            
        
    def run(self, frac, priority="random", seed=[], num_steps=1, nodes={}, g_sample={}, thresh=1):
        if priority=="random":
            self.random_immunization(frac)
        else:
            raise Exception("Invalid parameters")
        
        if not len(seed):
            seed = list(np.random.choice(self.g.nodes(), size=1, replace=False))
            
        # initialize sets of S/I/R/E nodes
        I = set(seed)
        S = set(self.g.nodes()).difference(I)
        R = set()
        E = set()
        t = 0
        
        #S->I transition
        SI = set(seed) 
        #I->R transition
        IR = set()   
        #R->E transition
        RE = set() 
        #E->I transition
        EI = set()    
        
        while True:
            # generator logic: yield current status every num_steps iterations
            if t % num_steps == 0:
                data=[t, len(S), len(I), len(R), len(E), len(SI), len(IR), len(RE), len(EI)]
                for i in range(len(self.keys)):
                    k=self.keys[i]
                    self.data[k].append(data[i])
            # stop when there are no infectious nodes left
            if not len(I):
                # normalize data
                self.data_normalized = {k: [v/self.N for v in self.data[k]] for k in self.data if k not in ["t"]}
                break

            SI = set()
            IR = set()
            RE = set()
            EI = set()
            
            # loop over neighbors of infectious nodes
            for i in set(I):
                # TRANSMISSION
                for j in S.intersection(self.g.neighbors(i)):
                    # Bernoulli sampling
                    if np.random.uniform() < self.BETA:
                        S.remove(j)
                        I.add(j)
                        SI.add(j)
                        
                # RECOVERY                
                if np.random.uniform() < self.GAMMA:
                    I.remove(i)
                    R.add(i)
                    IR.add(i)
               
                # EXPOSURE                
                if np.random.uniform() < 1-self.MU and i in R:
                    R.remove(i)
                    E.add(i)
                    RE.add(i)
               
                # RE-TRANSMISSION
                for j in E.intersection(self.g.neighbors(i)):
                    # immunization linear decrease of efficacy
                    if np.random.uniform() < 1-0.5*t*self.EPSILON:
                        E.remove(j)
                        I.add(j)
                        EI.add(j)
                    
            t += 1
            
   
    def getData(self):
        return self.data
    
    
    def getDataNormalized(self):
        return self.data_normalized

# END SIRE CLASS


# DEFINE PLOTTING FUNCTIONS

def plot_sir(plt, data, keys=["S", "I", "R"], names=['susceptible', 'infectious', 'recovered'], title="", flag=True):
    for k in keys:
        plt.plot(data[k])
        
    if flag:
        plt.title(title)
        plt.xlabel('time')
        plt.ylabel('# nodes')
        plt.legend(names,loc="best")

        
def plot_sire(plt, data, keys=["S", "I", "R", "E"], names=['susceptible', 'infectious', 'recovered', 'exposed'], title="", flag=True):
    for k in keys:
        plt.plot(data[k])
        
    if flag:
        plt.title(title)
        plt.xlabel('time')
        plt.ylabel('# nodes')
        plt.legend(names,loc="best")
        
# END PLOTTING FUNCTIONS

# DEFINE REWIRING OF GRAPH EDGES FUNCTION

def edge_rewire(g,p):
    g_s = g.copy() 
    N_s = g_s.order()
    
    #Sample a percentage p of random edges and nodes
    sampled_edges = random.sample(g_s.edges,int(p*N_s))
    sampled_nodes = random.sample(g_s.nodes,int(p*N_s))

    #rewire a percentage p of random edges
    for e in sampled_edges:
        #remove random edge
        g_s.remove_edge(*e)
        #add back the edge
        uv = random.sample(g_s.nodes,2)
        g_s.add_edge(*uv)

    #check
    print("Check." )
    print(nx.info(g_s))
    return g_s

# END OF REWIRING FUNCTION
# END
