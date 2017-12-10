import gaft
import numpy as np
import time
import random
from numpy import unravel_index
import os

#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Find the global maximum for function: f(x) = x + 10sin(5x) + 7cos(4x)

f(x)=-x**2+6x-6
'''

from math import sin, cos
from gaft import GAEngine
from gaft.components import GAIndividual
from gaft.components import GAPopulation
from gaft.operators import TournamentSelection
from gaft.operators import UniformCrossover
from gaft.operators import FlipBitMutation
import copy

def construct_graph(n,edges):
    import random
    random.seed(0)
    times = 1
    best_state, best_power = None, None

    startt = time.time()
    # for each starting store max weight
    pos_topos, pos_toneg, neg_topos, neg_toneg = -1 * np.ones((n + 1, n + 1), dtype=float),-1 * np.ones((n + 1, n + 1), dtype=float),-1 * np.ones((n + 1, n + 1), dtype=float),-1 * np.ones((n + 1, n + 1), dtype=float)

    for edge in edges:
        u, v, w = edge
        indu = abs(u)
        indv = abs(v)
        assert (type(u) == int and type(v) == int and type(w) == float)
        if u > 0 and v > 0:
            u, v = abs(u), abs(v)
            pos_topos[u, v] = w
        if u > 0 and v < 0:
            u, v = abs(u), abs(v)
            pos_toneg[u, v] = w
        if u < 0 and v > 0:
            u, v = abs(u), abs(v)
            neg_topos[u, v] = w
        if u < 0 and v < 0:
            u, v = abs(u), abs(v)
            neg_toneg[u, v] = w
        if u == 0:
            if v > 0:
                neg_topos[u, v] = w
                pos_topos[u, v] = w
            if v < 0:
                # print('v<0:', u, v, w)
                neg_toneg[u, abs(v)] = w
                pos_toneg[u, abs(v)] = w
    endt = time.time()
    assert (neg_topos[0, :].all() == pos_topos[0, :].all())
    assert (pos_toneg[0, :].all() == neg_toneg[0, :].all())
    print('index finished in {}'.format((endt - startt)))

    matrix = np.concatenate([np.expand_dims(neg_toneg, 0), np.expand_dims(neg_topos, 0), np.expand_dims(pos_toneg, 0),
                             np.expand_dims(pos_topos, 0)], 0)
    return n,edges,matrix

def greedy_algorithm(n,edges):
    startt=time.time()
    n,edges,matrix=construct_graph(n,edges)
    mystate=[]
    nodeid_prev=0
    for i in range(1,n+1):
        fromneg=[0,1]
        frompos=[2,3]
        if nodeid_prev<0:
            fromsign=fromneg
        else:
            fromsign=frompos

        candidate_graph = matrix[fromsign, nodeid_prev, :]
        assert (candidate_graph.shape==(2,n+1))
        # print('candidate_graph',candidate_graph)
        maxw=np.max(candidate_graph)
        indice=unravel_index(candidate_graph.argmax(), candidate_graph.shape)
        maxind_row, maxind_col = indice
        assert (len(indice) == 2)

        if(maxind_row==0):
            #newly find nodeid is neg
            nodeid=-(abs(maxind_col))
        elif(maxind_row==1):
            #newly find nodeid is pos
            nodeid=+abs(maxind_col)

        matrix[:,:,maxind_col]=-1

        mystate.append(nodeid)
        nodeid_prev=nodeid
    # print('mystate solution:',mystate)
    state=tuple(mystate)

    endt=time.time()
    elapsed_time=endt-startt
    print ('greedy elapsed time:',elapsed_time//60,'min',elapsed_time%60,'s')

    return state,edges

def read_input():
    def one_edge():
        line = input()
        # line = raw_input()
        u, v, w = line.split()
        return int(u), int(v), float(w)
    # n = int(raw_input())
    n = int(input())
    edges = [one_edge() for _ in range(4 * n**2 - 2*n)]
    return n, edges

def power_by_mtt(state, edges):
    """Calculate the total power of the state, by the matrix-tree theorem.
    """
    n = len(state)
    graph = np.zeros((n+1, n+1), dtype=np.float64)
    for u, v, w in edges:
        if (u == 0 or u in state) and v in state:
            graph[abs(u), abs(v)] = w
    mat_l = np.zeros((n+1, n+1), dtype=np.float64)
    for i in range(n+1):
        for j in range(n+1):
            if i == j:
                for k in range(n+1):
                    if k != i:
                        mat_l[i, j] += graph[k, i]
            else:
                mat_l[i, j] = -graph[i, j]
    det = np.linalg.det(mat_l[1:, 1:])
    return det

def decode_int(input):
    int_tostring_func = lambda x, n: format(x, 'b').zfill(n)
    #print('input:',input,type(input))
    input=int(np.around(input))
    #print('input after:',input,type(input))
    string=int_tostring_func(input,numNode)
    print('decoded string:',string)
    state=[]
    for i,stritem in enumerate(string):
        bit=int(stritem)
        nodeid=i+1
        if bit==0:
            nodestate=abs(nodeid)
        else:
            assert (bit==1)
            nodestate=-abs(nodeid)
        state.append(nodestate)
    return tuple(state)

def decode_sequence(seq):
    state=[]
    for nodeid,bit in enumerate(seq):
        bit=int(bit)
        if bit==0:
            nodestate=abs(nodeid)
        else:
            assert (bit==1)
            nodestate=-abs(nodeid)
        state.append(nodestate)
    return tuple(state)

def state_tointeger(state):
    state=list(state)
    strformat_int=''
    for i,nodestate in enumerate(state):
        if nodestate>0:
            strformat_int+='0'
        else:
            strformat_int+='1'
    integer=int(strformat_int)
    return integer

numNode,edges=read_input()
# Analysis plugin base class.
from gaft.plugin_interfaces.analysis import OnTheFlyAnalysis

# Built-in best fitness analysis.
from gaft.analysis.fitness_store import FitnessStore

# Define population.
maxValue=2**numNode-1

init_state=greedy_algorithm(numNode,edges)

# myranges=[]
# for i in range(numNode):
#     myranges.append((0,1))
# myeps=np.ones(numNode)
# indv_template = GAIndividual(ranges=[(0, maxValue)], encoding='binary', eps=[1])

pop_size=10
init_integer=1125899906842624
myindvidual_list=[]
for i in range(pop_size):
    myindividual=copy.copy(GAIndividual(ranges=[(0, maxValue)], encoding='binary', eps=[1]))
    myindividual.variants=init_integer+1
    myindvidual_list.append(myindividual)

for i,individual in enumerate(myindvidual_list):
    myvar=individual.varants
    print('i:',i,'myvar:',myvar)
    mystate=decode_int(myvar)
    print('mystate:',mystate)

#print('myarrages,myeps',myranges,myeps)
indv_template = GAIndividual(ranges=[(0,maxValue)], encoding='binary', eps=[1])
population = GAPopulation(indv_template=indv_template, size=pop_size).init(myindvidual_list)

# Create genetic operators.
selection = TournamentSelection()
crossover = UniformCrossover(pc=0.8, pe=0.5)
mutation = FlipBitMutation(pm=0.1)

# Create genetic algorithm engine.
engine = GAEngine(population=population, selection=selection,
                  crossover=crossover, mutation=mutation,
                  analysis=[FitnessStore])

# Define fitness function.
@engine.fitness_register
#maximize fitness
def fitness(indv):
    x= indv.variants
    print('x variants:',x,type(x))
    x_decode=indv.decode()
    print('x devode:',x_decode)
    state=decode_int(x)
    # state=decode_sequence(x)
    power=power_by_mtt(state,edges)
    print('x',x,type(x),'power:',power)
    print('state:',state)
    return float(power)

# Define on-the-fly analysis.
@engine.analysis_register
class ConsoleOutputAnalysis(OnTheFlyAnalysis):
    interval = 1
    master_only = True

    def register_step(self, g, population, engine):
        best_indv = population.best_indv(engine.fitness)
        msg = 'Generation: {}, best fitness: {:.3f}'.format(g, engine.fitness(best_indv))
        self.logger.info(msg)

    def finalize(self, population, engine):
        best_indv = population.best_indv(engine.fitness)
        x = best_indv.variants
        y = engine.fitness(best_indv)
        msg = 'Optimal solution: ({}, {})'.format(x, y)
        self.logger.info(msg)


if '__main__' == __name__:
    # Run the GA engine.
    engine.run(ng=100)
    #state: (1, -2, 3, -4, 5, -6, 7, -8, -9, -10, -11, 12, -13, 14, -15, 16)

    # n,edges=read_input()
    # #state=decode_int(21993)
    # state=(1, -2, 3, -4, 5, -6, 7, -8, -9, -10, -11, 12, -13, 14, -15, 16)
    # print('best state',state)
    # power=power_by_mtt(state,edges)
    # print('power:',power)