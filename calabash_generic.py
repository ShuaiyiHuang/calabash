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

# numNode=2
def decode_int(input):
    int_tostring_func = lambda x, n: format(x, 'b').zfill(n)
    print('input:',input,type(input))
    input=int(np.around(input))
    print('input after:',input,type(input))
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

numNode,edges=read_input()

# Analysis plugin base class.
from gaft.plugin_interfaces.analysis import OnTheFlyAnalysis

# Built-in best fitness analysis.
from gaft.analysis.fitness_store import FitnessStore

# Define population.
indv_template = GAIndividual(ranges=[[0, 2]], encoding='binary', eps=0.001)
population = GAPopulation(indv_template=indv_template, size=50).init()

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
    x, = indv.variants
    x_decode=indv.decode()
    state=decode_int(x)
    power=power_by_mtt(state,edges)
    print('x',x,type(x),'power:',power)
    print('state:',state)
    # return x + 10*sin(5*x) + 7*cos(4*x)
    #return -x**2+6*x-6
    return power

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
    # state=decode_int(2.99)
    # print('state:',state)