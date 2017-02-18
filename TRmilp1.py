# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
from gurobipy import *
import time

INF = np.inf
Nmax = 10 # max number of regenerator circuits per regenerator node
np.random.seed(0) # set random seed
bigM1 = 10**4 # big number
bigM2 = 10**5 
bigM3 = 2*10**6 
G = 10 # guardband

class Network(object):
    '''The network 
    '''
    
    def __init__(self, cost_matrix):
        '''Initialize the network topology'''
        self.cost_matrix = cost_matrix
        self.connectivity_matrix = 1-np.isinf(self.cost_matrix)
        self.n_nodes = self.cost_matrix.shape[0]
        self.nodes = list(range(self.n_nodes))
        self.node_pairs = [(i, j) for i in range(self.n_nodes) 
            for j in range(self.n_nodes) if i!=j]
        self.n_node_pairs = len(self.node_pairs)
        links = [[i, j, self.cost_matrix[i,j]] 
            for i in range(self.n_nodes)
            for j in range(self.n_nodes)
            if i!=j and 1-np.isinf(self.cost_matrix[i,j])]
        ids = [i for i in range(len(links))]
        src = [links[i][0] for i in range(len(links))]
        dst = [links[i][1] for i in range(len(links))]
        length = [int(links[i][2]) for i in range(len(links))]
        self.n_links = len(links)
        self.links = pd.DataFrame({'id': ids, 'source': src, 
                                   'destination': dst, 'length':length})
        self.links = self.links[['id', 'source', 'destination', 'length']]
    
    def create_demands(self, n_demands, low=30, high=400):
        '''Create demands
        '''
        demand_pairs_id = np.random.choice(self.n_node_pairs, n_demands)
        demand_pairs_id = sorted(demand_pairs_id)
        demand_pairs = [self.node_pairs[i] for i in demand_pairs_id]
        data_rates = np.random.randint(low, high, n_demands)
        src = [x[0] for x in demand_pairs]
        dst = [x[1] for x in demand_pairs]
        ids = [i for i in range(n_demands)]
        demands = pd.DataFrame({'id': ids, 'source':src, 'destination':dst, 
                                'data_rates': data_rates})
        demands = demands[['id', 'source', 'destination', 'data_rates']]

        tr = [3651-1.25*data_rates[i] for i in range(n_demands)]
        demands['TR'] = tr

        return demands
    
    def solve(self, demands, **kwargs):
        '''Formulate and solve
        '''
        n_demands = demands.shape[0]
        
        # define supply 
        supply = np.zeros((self.n_nodes, n_demands))
        for n in range(self.n_nodes):
            for d in range(n_demands):
                if demands.source[d]==n:
                    supply[n, d] = -1

        for n in range(self.n_nodes):
            for d in range(n_demands):
                if demands.destination[d]==n:
                    supply[n, d] = 1
        
        tic = time.clock()
        model = Model('TR')
        
        # define variables
        UsageL = {} # if demand d uses link l
        for l in self.links.id:
            for d in demands.id:
                UsageL[l, d] = model.addVar(vtype=GRB.BINARY)
                
        Fstart = {} # the start frequency of demand d
        for d in demands.id:
            Fstart[d] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=bigM1)
            
        Delta = {} # order between demands
        for d1 in demands.id:
            for d2 in demands.id:
                if d1!=d2:
                    Delta[d1, d2] = model.addVar(vtype=GRB.BINARY)
                    
        U = {} # U[a, b] = UsageL[a,b]*Ynode[a,b]
        for l in self.links.id:
            for d in demands.id:
                U[l, d] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=bigM1)
                
        Ire = {} # 
        for n in self.nodes:
            for d in demands.id:
                Ire[n, d] = model.addVar(vtype=GRB.BINARY)
                
        III = {} # 
        for n in self.nodes:
            for d in demands.id:
                III[n, d] = model.addVar(vtype=GRB.BINARY)
                
        I = {}
        for n in self.nodes:
            I[n] = model.addVar(vtype=GRB.BINARY)
                
        NNN = {}
        for n in self.nodes:
            NNN[n] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=10)
            
        X = {}
        for l in self.links.id:
            for d in demands.id:
                X[l, d] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=bigM1)
                
        Ynode = {}
        for n in self.nodes:
            for d in range(n_demands):
                Ynode[n, d] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=bigM1)
                
        Total = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=self.n_nodes)
        
        c = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=bigM2)
        
        model.update()
        
        # define constraints
        model.addConstr(Total==quicksum(I[n] for n in self.nodes))
        
        for d in demands.id:
            model.addConstr(c>=Fstart[d]+demands.data_rates[d])
            
        # flow conservation
        for n in self.nodes:
            for d in demands.id:
                model.addConstr(quicksum(UsageL[l, d] for l in self.links.id
                                         if self.links.source[l]==n)-
                                quicksum(UsageL[l, d] for l in self.links.id
                                         if self.links.destination[l]==n)
                                == supply[n, d])
                                
        for d1 in demands.id:
            for d2 in demands.id:
                if d1!=d2:
                    model.addConstr(Delta[d1, d2]+Delta[d2, d1]==1)
                    
        for d1 in demands.id:
            for d2 in demands.id:
                for l in self.links.id:
                    if d1!=d2:
                        model.addConstr(Fstart[d1]-Fstart[d2]<=
                                        bigM3*(3-Delta[d1, d2]-
                                               UsageL[l, d1]-UsageL[l, d2]))
                    
        for d1 in demands.id:
            for d2 in demands.id:
                for l in self.links.id:
                    if d1!=d2:
                        model.addConstr(Fstart[d1]-Fstart[d2]+
                                        demands.data_rates[d1]+G<=
                                        bigM3*(3-Delta[d1, d2]-UsageL[l, d1]-
                                        UsageL[l, d2]))
                    
        for l in self.links.id:
            for d in demands.id:
                model.addConstr(U[l, d]<=UsageL[l, d]*demands.TR[d])
                model.addConstr(U[l, d]<=Ynode[self.links.source[l], d])
                model.addConstr(Ynode[self.links.source[l], d]-U[l, d]<=
                                demands.TR[d]*(1-UsageL[l, d]))
                
        for n in self.nodes:
            for d in demands.id:
                model.addConstr(Ynode[n, d]==
                                quicksum(X[l, d]+UsageL[l, d]*
                                          self.links.length[l] 
                                          for l in self.links.id 
                                          if self.links.destination[l]==n))
                
        for n in self.nodes:
            for d in demands.id:
                model.addConstr(III[n, d]==1-Ire[n, d])
                
        for l in self.links.id:
            for d in demands.id:
                model.addConstr(X[l, d]<=bigM1*III[self.links.source[l], d])
                model.addConstr(X[l, d]<=U[l, d])
                model.addConstr(X[l, d]>=U[l, d]-
                                bigM1*(1-III[self.links.source[l], d]))
                model.addConstr(X[l, d]>=0)
        
        for n in self.nodes:
            model.addConstr(NNN[n]==quicksum(Ire[n, d] for d in demands.id))
            model.addConstr(I[n]*Nmax>=NNN[n])
            
        # objective
        model.setObjective(c+Total, GRB.MINIMIZE)
            
        if len(kwargs):
            for key, value in kwargs.items():
                setattr(model.params, key, value)
                
        model.optimize()
        
        toc = time.clock()
        self.runtime = toc-tic
        
        return model

#model = Model('TRmilp')

if __name__=='__main__':
    network_cost = np.array([[INF, 5, INF, INF, INF, 6], 
                             [5, INF, 4, INF, 5, 3],
                             [INF, 4, INF, 5, 3, INF],
                             [INF, INF, 5, INF, 6, INF],
                             [INF, 5, 3, 6, INF, 4], 
                             [6, 3, INF, INF, 4, INF]])
    sn = Network(network_cost)
    n_demands = 15
    demands = sn.create_demands(n_demands)
    model = sn.solve(demands, mipfocus=1, timelimit=20, method=2, mipgap=0.01)
    