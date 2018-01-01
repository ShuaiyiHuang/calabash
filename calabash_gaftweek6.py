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
from gaft.operators import TournamentSelection, RouletteWheelSelection
from gaft.operators import UniformCrossover
from gaft.operators import FlipBitMutation
import argparse

graph = None


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
    print('path:', path)
    if not os.path.exists(path):

        os.mkdir(path)
    with open(os.path.join(path,filename),'w') as f:
        string_states = [str(nodestate) for nodestate in list(best_states)]

        string_states = ' '.join(string_states)
        f.write(string_states)
        f.close()

def decode_sequence(seq):
    # 0 :positive;1:negative
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

def construct_graph(input_path):
    import random
    random.seed(0)

    n, edges = read_input(input_path)

    startt = time.time()
    # for each starting store max weight
    pos_topos, pos_toneg, neg_topos, neg_toneg = np.zeros((n + 1, n + 1), dtype=float),np.zeros((n + 1, n + 1), dtype=float),np.zeros((n + 1, n + 1), dtype=float),np.zeros((n + 1, n + 1), dtype=float)

    for edge in edges:
        u, v, w = edge
        indu = abs(u)
        indv = abs(v)
        assert (type(u) == int and type(v) == int and type(w) == float)
        if u > 0 and v > 0:
            pos_topos[indu, indv] = w
        if u > 0 and v < 0:
            pos_toneg[indu, indv] = w
        if u < 0 and v > 0:
            neg_topos[indu, indv] = w
        if u < 0 and v < 0:
            neg_toneg[indu, indv] = w
        if u == 0:
            if v > 0:
                neg_topos[indu, indv] = w
                pos_topos[indu, indv] = w
            if v < 0:
                neg_toneg[indu, indv] = w
                pos_toneg[indu, indv] = w

    assert (neg_topos[0, :].all() == pos_topos[0, :].all())
    assert (pos_toneg[0, :].all() == neg_toneg[0, :].all())

    # graph = np.concatenate([np.expand_dims(neg_toneg, 0), np.expand_dims(neg_topos, 0), np.expand_dims(pos_toneg, 0),
    #                          np.expand_dims(pos_topos, 0)], 0)
    graph_frompos=np.concatenate((pos_topos,pos_toneg),1)
    assert (graph_frompos.shape==(n+1,2*(n+1)))
    graph_fromneg=np.concatenate((neg_topos,neg_toneg),1)
    assert(graph_fromneg.shape==(n+1,2*(n+1)))
    graph=np.concatenate((graph_frompos,graph_fromneg),0)
    assert (graph.shape==(2*(n+1),2*(n+1)))

    endt = time.time()
    print('index finished in {}'.format((endt - startt)))

    return n,edges,graph


def power_by_mtt_fastgraph(states, graph):
    n = len(states)
    #pos then neg

    states_ind=[0]
    for i,nodestate in enumerate(states):
        #pos 0,neg 1
        sign=1 if nodestate<0 else 0
        ind_insubgraph=abs(nodestate)+sign*(n+1)
        states_ind.append(ind_insubgraph)

    subgraph = graph[states_ind]
    subgraph = subgraph[:,states_ind]

    colum_sum_vec=np.sum(subgraph,0)
    new_digvalue=colum_sum_vec
    reverse_graph=-subgraph

    indices_diag=np.diag_indices(n+1)

    mat_l=reverse_graph
    mat_l[indices_diag]=new_digvalue

    det = np.linalg.det(mat_l[1:, 1:])
    return det

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='tune calabash')
    parser.add_argument('--outputpath', type=str, default='./output',
                        help='output path')
    parser.add_argument('--inputpath', type=str, default='./input',
                        help='output path')
    parser.add_argument('--filename', type=str, default='6',
                        help='filename')

    parser.add_argument('--tselect', type=int, default=1,
                        help='1:TournamentSelection,0:RouletteWheelSelection')
    parser.add_argument('--popsize', type=int, default=1000,
                        help='size of population')
    parser.add_argument('--pc', type=float, default=0.8,
                        help='probability of cross over')
    parser.add_argument('--pe', type=float, default=0.9,
                        help='probability of exchange')
    parser.add_argument('--pm', type=float, default=0.2,
                        help='probability of mutation')
    parser.add_argument('--numEpoch', type=int, default=800,
                        help='num of generation')

    parser.add_argument('--rseed', type=int, default=30,
                        help='random.seed')
    parser.add_argument('--npseed', type=int, default=32,
                        help='np.random.seed')

    args = parser.parse_args()
    print(args)
    #path args
    inputpath = args.inputpath
    outputpath = args.outputpath
    filename=args.filename

    #model tuning args
    popsize = args.popsize
    pc = args.pc
    pe = args.pe
    pm = args.pm
    numEpoch = args.numEpoch
    rseed=args.rseed
    npseed=args.npseed
    tselect=args.tselect

    random.seed(rseed)
    np.random.seed(npseed)
    inputdir= os.path.join(inputpath,filename)

    n,edges,graph=construct_graph(inputdir)
    maxValue = 2 ** n-1

    indv_template = GAIndividual(ranges=[(0, maxValue)], encoding='binary', eps=[1])
    population = GAPopulation(indv_template=indv_template, size=popsize).init()

    # Create genetic operators.
    # selection = TournamentSelection()#RouletteWheelSelection()
    if tselect:
        print('Tournament selection')
        selection = TournamentSelection()
    else:
        print('RouletteWheelSelection')
        selection = RouletteWheelSelection()

    crossover = UniformCrossover(pc=pc, pe=pe)
    mutation = FlipBitMutation(pm=pm)

    # Create genetic algorithm engine.
    engine = GAEngine(population=population, selection=selection,
                      crossover=crossover, mutation=mutation,
                      analysis=[FitnessStore])


    @engine.fitness_register
    def fitness(indv):
        # print('x variants',indv.variants)
        x_decode = indv.chromsome
        state = decode_sequence(x_decode)
        power = power_by_mtt_fastgraph(state, graph)
        return float(power)


    # Define on-the-fly analysis.
    @engine.analysis_register
    class ConsoleOutputAnalysis(OnTheFlyAnalysis):
        interval = 1
        master_only = True

        def register_step(self, g, population, engine):
            best_indv = population.best_indv(engine.fitness)
            f = engine.fitness(best_indv)
            msg = 'Generation: {}, best fitness: {}'.format(g, f)
            self.logger.info(msg)

        def finalize(self, population, engine):
            best_indv = population.best_indv(engine.fitness)
            x = best_indv.variants
            y = engine.fitness(best_indv)
            msg = 'Optimal solution: ({}, {})'.format(x, y)
            self.logger.info(msg)

    startt=time.time()
    engine.run(ng=numEpoch)
    endt=time.time()
    print('Runing cost time:',(endt-startt)//60,'min',(endt-startt),'s')

    best_indv = population.best_indv(engine.fitness)
    x_decode = best_indv.chromsome
    best_state=decode_sequence(x_decode)
    best_power=power_by_mtt_fastgraph(best_state,graph)
    print('best state:',best_state)
    print('best power:',best_power)

    #score
    score=int(1e9 + 1e6*np.log(best_power))
    print('score:',score)

    write_output(best_state, outputpath, filename)




