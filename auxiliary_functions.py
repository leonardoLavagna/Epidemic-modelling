class SIR:
    def __init__(self, graph, recovery, beta=beta, gamma=gamma):
        self.g = graph.copy()   
        self.N = g.order()
        self.BETA = beta  
        self.GAMMA = gamma      
        self.keys=['t', 'S', 'I', 'R', 'SI', 'IR']
        self.data = { k: [] for k in self.keys }
        self.data_normalized = { k: [] for k in self.keys}
        self.RECOVERY = recovery
        
    def random_immunization(self, frac):
        n=int(self.N * frac)
        immune = list(np.random.choice(self.g.nodes(), size=n, replace=False))
        for node in immune:
            self.g.remove_node(node)
            
    def prioritized_immunization(self, frac, flag="cc"):
        '''
        apply immunization ordering for higher degree centrality
        '''
        n=int(self.N * frac)
        
        degree_dict=self.g.degree()
        if flag=="cc":
            degree_dict=nx.closeness_centrality(g)
        elif flag=="bc":
            degree_dict=nx.betweenness_centrality(g)
        elif flag=="c":
            degree_dict=nx.betweenness_centrality(g)
            
        degree=sorted( [(k, degree_dict[k]) for k in degree_dict], key=lambda u:u[1], reverse=True )
        immune=[d[0] for d in degree[:n]]
        for node in immune:
            self.g.remove_node(node)

            
    def sample_immunization(self, frac, g_sample, thresh):
        '''
        degree immunization and random immunization (if r > k)  
        '''
        n=int( self.N * frac )
        degree_dict_tot=g_sample.degree()
        ## remove nodes with degree 1 (better to sample random)
        degree_dict={k: int(degree_dict_tot[k]) for k in degree_dict_tot if degree_dict_tot[k] > thresh}
        ## sort
        degree=sorted( [(k, degree_dict[k]) for k in degree_dict], key=lambda u:u[1], reverse=True )
        immune=[d[0] for d in degree[:n]]
        for node in immune:
            self.g.remove_node(node)
        
        x = (n - len(degree))
        if x > 0:
            x = x / self.N
            self.random_immunization(x)
        
    def run(self, seed=[], num_steps=1, priority="", frac=0.0, nodes={}, g_sample={}, thresh=1):
        if priority=="random":
            self.random_immunization(frac)
        elif priority=="degree":
            self.prioritized_immunization(frac)
        elif priority=="sample_immunization":
            self.sample_immunization(frac, g_sample, thresh)
            
        if not len(seed):
            seed = list(np.random.choice(self.g.nodes(), size=1, replace=False))
            
        # initialize sets of S/I/R nodes
        I = set(seed)
        S = set(self.g.nodes()).difference(I)
        R = set()
        t = 0
        
        SI = set(seed) #S->I transition
        IR = set()     #I->R transition
        
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
                if self.RECOVERY == "gaussian":
                    if np.random.normal(0,1,None) < self.GAMMA:
                        I.remove(i)
                        R.add(i)
                        IR.add(i)
                # power law with a=2
                elif self.RECOVERY == "power" :
                    if np.random.power(2) < self.GAMMA:
                        I.remove(i)
                        R.add(i)
                        IR.add(i)
                # uniform    
                elif self.RECOVERY == "uniform" :
                    if np.random.uniform() < self.GAMMA:
                        I.remove(i)
                        R.add(i)
                        IR.add(i)
                    
            t += 1
            
    def getSeed(self):
        return self.seed
    def getData(self):
        return self.data
    def getDataNormalized(self):
        return self.data_normalized


def plot_sir(plt, data, keys=["S", "I", "R"], names=['susceptible', 'infectious', 'recovered'], title="", flag=True):
    for k in keys:
        plt.plot(data[k])
        
    if flag:
        plt.title(title)
        plt.xlabel('time')
        plt.ylabel('# nodes')
        plt.legend(names,loc="best")