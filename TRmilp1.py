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

        tr = [(3651-1.25*data_rates[i])/100 for i in range(n_demands)]
        demands['TR'] = tr

        return demands
    
    def solve_all(self, demands, **kwargs):
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
                UsageL[l, d] = model.addVar(vtype=GRB.BINARY, 
                    name='UsageL_{}_{}'.format(l, d))
                
        Fstart = {} # the start frequency of demand d
        for d in demands.id:
            Fstart[d] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=bigM1, 
                  name='Fstart_{}'.format(d))
            
        Delta = {} # order between demands
        for d1 in demands.id:
            for d2 in demands.id:
                if d1!=d2:
                    Delta[d1, d2] = model.addVar(vtype=GRB.BINARY, 
                         name='Delta_{}_{}'.format(d1, d2))
                    
        U = {} # U[a, b] = UsageL[a,b]*Ynode[a,b]
        for l in self.links.id:
            for d in demands.id:
                U[l, d] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=bigM1, 
                 name='U_{}_{}'.format(l, d))
                
        Ire = {} # 
        for n in self.nodes:
            for d in demands.id:
                Ire[n, d] = model.addVar(vtype=GRB.BINARY, 
                   name='Ire_{}_{}'.format(n, d))
                
        III = {} # 
        for n in self.nodes:
            for d in demands.id:
                III[n, d] = model.addVar(vtype=GRB.BINARY, 
                   name='III_{}_{}'.format(n, d))
                
        I = {}
        for n in self.nodes:
            I[n] = model.addVar(vtype=GRB.BINARY, name='I_{}'.format(n))
                
        NNN = {}
        for n in self.nodes:
            NNN[n] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=10, 
               name='NNN_{}'.format(n))
            
        X = {}
        for l in self.links.id:
            for d in demands.id:
                X[l, d] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=bigM1, 
                 name='X_{}_{}'.format(l, d))
                
        Ynode = {}
        for n in self.nodes:
            for d in range(n_demands):
                Ynode[n, d] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, 
                     ub=bigM1, name='Ynode_{}_{}'.format(n, d))
                
        Total = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=self.n_nodes, 
                             name='Total')
        
        c = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=bigM2, name='c')
        
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
            
        # set gurobi parameters
        if len(kwargs):
            for key, value in kwargs.items():
                try:
                    setattr(model.params, key, value)
                except:
                    pass
                
        toc = time.clock()
        self.model_time = toc-tic
                
        model.optimize()
        
        toc2 = time.clock()
        self.solve_time = toc2-toc
        
        solutions = {}
        solutions['UsageL'] = UsageL
        solutions['Fstart'] = Fstart
        solutions['Delta'] = Delta
        solutions['U'] = U
        solutions['Ire'] = Ire
        solutions['III'] = III
        solutions['I'] = I
        solutions['NNN'] = NNN
        solutions['X'] = X
        solutions['Ynode'] = Ynode
        solutions['Total'] = Total
        solutions['c'] = c
        
        # save files
        if len(kwargs):
            for key, value in kwargs.items():
                if key=='write':
                    if type(value) is list:
                        for i in value:
                            try:
                                model.write(i)
                            except:
                                pass
                    elif type(value) is str:
                        try:
                            model.write(value)
                        except:
                            pass
                        
        # construct solutions for UsageL and Delta
        UsageLx = {}
        for l in self.links.id:
            for d in demands.id:
                UsageLx[l, d] = UsageL[l, d].x
                       
        Deltax = {}
        for d1 in demands.id:
            for d2 in demands.id:
                if d1!=d2:
                    Deltax[d1, d2] = Delta[d1, d2].x
        
        return model, solutions, UsageLx, Deltax

    def solve_partial(self, demands, previous_solutions, **kwargs):
        '''Formulate and solve iteratively
        previous_solutions is dict, contains:
            - UsageL from the previous solve, dict
            - Delta from the previous solve, dict
            - demands_added new demands that will be allocated this time, list
            - demands_fixed old demands allocated previously, list, 
                can be empty
            we should make sure that UsageL and Delta contain solutions for 
            all the demands in demands_fixed
        '''
        # process input data
        demands_added = previous_solutions['demands_added'] # new demands
        demands_fixed = previous_solutions['demands_fixed'] # old demands
        UsageLx = previous_solutions['UsageL']
        Deltax = previous_solutions['Delta']
        demands_all = demands_added+demands_fixed 
        n_demands = len(demands_all)
        demands = demands.loc[demands.id.isin(demands_all)]
        

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
                if d in demands_fixed:
                    UsageL[l, d] = model.addVar(vtype=GRB.BINARY, 
                        name='UsageL_{}_{}'.format(l, d), 
                        lb=UsageLx[l, d], ub=UsageLx[l, d])
                elif d in demands_added:
                    UsageL[l, d] = model.addVar(vtype=GRB.BINARY, 
                        name='UsageL_{}_{}'.format(l, d))
                
        Fstart = {} # the start frequency of demand d
        for d in demands.id:
            Fstart[d] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=bigM1, 
                  name='Fstart_{}'.format(d))
            
        Delta = {} # order between demands
        for d1 in demands.id:
            for d2 in demands.id:
                if (d1!=d2) and (d1 in demands_fixed) and \
                   (d2 in demands_fixed):
                    Delta[d1, d2] = model.addVar(vtype=GRB.BINARY, 
                         name='Delta_{}_{}'.format(d1, d2), 
                         lb=Deltax[d1, d2], ub=Deltax[d1, d2])
                elif d1!=d2:
                    Delta[d1, d2] = model.addVar(vtype=GRB.BINARY, 
                         name='Delta_{}_{}'.format(d1, d2))
                    
        U = {} # U[a, b] = UsageL[a,b]*Ynode[a,b]
        for l in self.links.id:
            for d in demands.id:
                U[l, d] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=bigM1, 
                 name='U_{}_{}'.format(l, d))
                
        Ire = {} # 
        for n in self.nodes:
            for d in demands.id:
                Ire[n, d] = model.addVar(vtype=GRB.BINARY, 
                   name='Ire_{}_{}'.format(n, d))
                
        III = {} # 
        for n in self.nodes:
            for d in demands.id:
                III[n, d] = model.addVar(vtype=GRB.BINARY, 
                   name='III_{}_{}'.format(n, d))
                
        I = {}
        for n in self.nodes:
            I[n] = model.addVar(vtype=GRB.BINARY, name='I_{}'.format(n))
                
        NNN = {}
        for n in self.nodes:
            NNN[n] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=10, 
               name='NNN_{}'.format(n))
            
        X = {}
        for l in self.links.id:
            for d in demands.id:
                X[l, d] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=bigM1, 
                 name='X_{}_{}'.format(l, d))
                
        Ynode = {}
        for n in self.nodes:
            for d in range(n_demands):
                Ynode[n, d] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, 
                     ub=bigM1, name='Ynode_{}_{}'.format(n, d))
                
        Total = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=self.n_nodes, 
                             name='Total')
        
        c = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=bigM2, name='c')
        
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
            
        # set gurobi parameters
        if len(kwargs):
            for key, value in kwargs.items():
                try:
                    setattr(model.params, key, value)
                except:
                    pass
                
        toc = time.clock()
        self.model_time = toc-tic
                
        model.optimize()
        
        toc2 = time.clock()
        self.solve_time = toc2-toc
        
        solutions = {}
        solutions['UsageL'] = UsageL
        solutions['Fstart'] = Fstart
        solutions['Delta'] = Delta
        solutions['U'] = U
        solutions['Ire'] = Ire
        solutions['III'] = III
        solutions['I'] = I
        solutions['NNN'] = NNN
        solutions['X'] = X
        solutions['Ynode'] = Ynode
        solutions['Total'] = Total
        solutions['c'] = c
        
        # save files
        if len(kwargs):
            for key, value in kwargs.items():
                if key=='write':
                    if type(value) is list:
                        for i in value:
                            try:
                                model.write(i)
                            except:
                                pass
                    elif type(value) is str:
                        try:
                            model.write(value)
                        except:
                            pass
                        
        # construct solutions for UsageL and Delta
        UsageLx = {}
        for l in self.links.id:
            for d in demands.id:
                UsageLx[l, d] = UsageL[l, d].x
                       
        Deltax = {}
        for d1 in demands.id:
            for d2 in demands.id:
                if d1!=d2:
                    Deltax[d1, d2] = Delta[d1, d2].x
        
        
        return model, solutions, UsageLx, Deltax
    
    def scheduler(self, idx, iteration_history):
        '''Generate demands_fixed and demands_added according to history
        idx is the index of the next step, idx 
        iteration_history contains:
            - step id
            - demands_fixed
            - demands_added
            - solutions
            - UsageLx
            - Deltax
        '''
        
    
    def iterate(self, demands, random_state=0, **kwargs):
        '''Iterate 
        
        '''
        
        np.random.seed(random_state)
        demands_id = demands.id.as_matrix()
        np.random.shuffle(demands_id)
        
        iteration_history = {}
        iteration_history[0, 'step_id'] = 0
        iteration_history[0, 'demands_fixed'] = []
        iteration_history[0, 'demands_added'] = []
        iteration_history[0, 'solutions'] = []
        iteration_history[0, 'UsageLx'] = []
        iteration_history[0, 'Deltax'] = []
        
        
        
        return iteration_history
        

if __name__=='__main__':
#    network_cost = np.array([[INF, 5, INF, INF, INF, 6], 
#                             [5, INF, 4, INF, 5, 3],
#                             [INF, 4, INF, 5, 3, INF],
#                             [INF, INF, 5, INF, 6, INF],
#                             [INF, 5, 3, 6, INF, 4], 
#                             [6, 3, INF, INF, 4, INF]])
    network_cost = pd.read_csv('networkDT.csv', header=None)/100
    network_cost = network_cost.as_matrix()
    sn = Network(network_cost)
    n_demands = 10
    demands = sn.create_demands(n_demands, low=40, high=100)
#    model, solutions, UsageLx, Deltax = \
#        sn.solve_all(demands, mipfocus=1, timelimit=1, method=2, 
#                         mipgap=0.01, write=['solution.sol', 'test.mps'])
#    
#    previous_solutions = {}
#    previous_solutions['demands_added'] = [7, 8, 9]
#    previous_solutions['demands_fixed'] = [0, 1, 2, 3, 4, 5, 6]
#    previous_solutions['UsageL'] = UsageLx
#    previous_solutions['Delta'] = Deltax
#    model, solutions, UsageLx, Deltax = sn.solve_partial(demands, previous_solutions)
    
    iteration_history = sn.iterate(demands)