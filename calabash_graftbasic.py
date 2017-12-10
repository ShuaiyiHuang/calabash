import gaft
import numpy as np
import time
import random
from numpy import unravel_index
import os
# Analysis plugin base class.
from gaft.plugin_interfaces.analysis import OnTheFlyAnalysis
# Built-in best fitness analysis.
from gaft.analysis.fitness_store import FitnessStore
from gaft import GAEngine
from gaft.components import GAIndividual
from gaft.components import GAPopulation
from gaft.operators import TournamentSelection
from gaft.operators import UniformCrossover
from gaft.operators import FlipBitMutation

def read_input(input_path):
    if not os.path.exists(input_path):
        print('Error:input file not exitst!')
        return None,None

    edges=[]
    with open(input_path) as f:
        lines = f.read().splitlines()
        line0=lines[0]
        print('numNode:',line0)
        n=int(line0)
        for i,line in enumerate(lines[1:]):
            u, v, w = line.split()
            one_edge=(int(u), int(v), float(w))
            edges.append(one_edge)
        f.close()
    return n,edges

def write_output(best_states,path,filename):
    if not os.path.exists(path):
        os.mkdir(path)
    with open(os.path.join(path,filename),'w') as f:
        string_states = [str(nodestate) for nodestate in list(best_states)]

        string_states = ' '.join(string_states)
        f.write(string_states)
        f.close()

def decode_sequence(seq):
    state=[]
    for i,bit in enumerate(seq):
        nodeid = i + 1
        bit=int(bit)
        if bit==0:
            nodestate=abs(nodeid)
        else:
            assert (bit==1)
            nodestate=-abs(nodeid)
        state.append(nodestate)
    return tuple(state)

def power_by_mtt_fast(state,edges):
    """Calculate the total power of the state, by the matrix-tree theorem with vectorized code.
    """
    n = len(state)
    graph = np.zeros((n+1, n+1), dtype=np.float64)
    for u, v, w in edges:
        if (u == 0 or u in state) and v in state:
            graph[abs(u), abs(v)] = w
    diag=np.diagonal(graph)
    colum_sum_vec=np.sum(graph,0)
    new_digvalue=colum_sum_vec-diag
    reverse_graph=-graph

    indices_diag=np.diag_indices(n+1)

    mat_l=reverse_graph
    mat_l[indices_diag]=new_digvalue

    det = np.linalg.det(mat_l[1:, 1:])
    return det


if __name__ == '__main__':
    random.seed(10)

    input_path='./input/1'
    output_path = './output'
    filename='small_gaft'
    n,edges=read_input(input_path)
    maxValue = 2 ** n
    pop_size=600

    indv_template = GAIndividual(ranges=[(0, maxValue)], encoding='binary', eps=[1])
    population = GAPopulation(indv_template=indv_template, size=pop_size).init()

    # Create genetic operators.
    selection = TournamentSelection()
    crossover = UniformCrossover(pc=0.9, pe=0.9)
    mutation = FlipBitMutation(pm=0.1)

    # Create genetic algorithm engine.
    engine = GAEngine(population=population, selection=selection,
                      crossover=crossover, mutation=mutation,
                      analysis=[FitnessStore])


    @engine.fitness_register
    def fitness(indv):
        x_decode = indv.chromsome

        state = decode_sequence(x_decode)
        power = power_by_mtt_fast(state, edges)
        # print('power:',indv.variants,indv.chromsome,power)
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


    engine.run(ng=10)

    best_indv = population.best_indv(engine.fitness)
    x_decode = best_indv.chromsome
    best_state=decode_sequence(x_decode)
    best_power=power_by_mtt_fast(best_state,edges)
    print('best state:',best_state)
    print('best power:',best_power)

    write_output(best_state,output_path,filename)




