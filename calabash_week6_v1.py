
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

from calabash_utils import *

graph = None



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='tune calabash')
    parser.add_argument('--outputpath', type=str, default='./output',
                        help='output path')
    parser.add_argument('--inputpath', type=str, default='./input',
                        help='output path')
    parser.add_argument('--filename', type=str, default='6',
                        help='filename')

    parser.add_argument('--tselect', type=int, default=0,
                        help='1:TournamentSelection,0:RouletteWheelSelection')
    parser.add_argument('--popsize', type=int, default=2000,
                        help='size of population')
    parser.add_argument('--pc', type=float, default=0.8,
                        help='probability of cross over')
    parser.add_argument('--pe', type=float, default=0.9,
                        help='probability of exchange')
    parser.add_argument('--pm', type=float, default=0.3,
                        help='probability of mutation')
    parser.add_argument('--numEpoch', type=int, default=500,
                        help='num of generation')

    parser.add_argument('--rseed', type=int, default=87678,
                        help='random.seed')
    parser.add_argument('--npseed', type=int, default=132,
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

    # indv_template = GAIndividual(ranges=[(0, maxValue)], encoding='binary', eps=[1])

    ##################Initialize approach1######################################
    length_part1 = int(np.floor(n * 0.5))
    maxValue_part1=2**length_part1

    length_part2=n-length_part1
    maxValue_part2=2**length_part2
    print('maxValue part1:',maxValue_part1)
    print('maxValue part2:',maxValue_part2)
    print('length part1 part2:',length_part1,length_part2)

    var1,var2=greedy_initialize_variants(n,edges,length_part1,length_part2,graph)
    indv_template = GAIndividual(ranges=[(0, maxValue_part1),(0,maxValue_part2)], encoding='binary', eps=[1,1])

    indv_template.variants=[var1,var2]
    indv_template.chromsome=indv_template.encode()


    # var=greedy_initialize_one_variants(n,edges,graph)
    # indv_template = GAIndividual(ranges=[(0, maxValue)], encoding='binary', eps=[1])
    # indv_template.variants=[var]
    # indv_template.chromsome=indv_template.encode()
    #####################Initialize approach2##############################
    # ranges_list = []
    # eps_list = []
    # for i in range(n):
    #     ranges_list.append((0,2))
    #     eps_list.append(1)
    # print(len(ranges_list),ranges_list)
    # print(len(eps_list),eps_list)
    # assert (len(ranges_list)==len(eps_list) and len(eps_list)==n)
    # indv_template = GAIndividual(ranges=ranges_list, encoding='binary', eps=eps_list)
    indv_list=[]
    for i in range(popsize):
        indv_list.append(indv_template.clone())

    population = GAPopulation(indv_template=indv_template, size=popsize).init(indv_list)

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
        # print('x variants',indv.variants,type(indv.variants),len(indv.variants))
        x_decode = indv.chromsome
        # floatpart1,floatpart2=indv.variants
        # binary_part1,binary_part2=binarize(floatpart1,eps=1,length=length_part1),binarize(floatpart2,eps=1,length=length_part2)
        # x_decode=binary_part1+binary_part2
        # assert (len(binary_part1)==length_part1)
        # assert (len(binary_part2)==length_part2)
        # assert (len(x_decode)==n)
        # if len(chromsome)!=n:
        #     print('len chromsome:',len(chromsome))

        # if len(chromsome)!=len(x_decode):
        #     print('not consistant:',len(chromsome),len(x_decode))
        # if chromsome!=x_decode:
        #     print('not equal')
        #     print('chrom:',chromsome)
        #     print('x_dec:',x_decode)
        # x_decode=indv.variants
        # print('x_decode:',x_decode)
        assert (len(x_decode)==n)
        state = decode_sequence(x_decode)
        power = power_by_mtt_fastgraph(state, graph)
        # print('power',power)
        return float(power)

    # def get_x_decode(indv):
    #     floatpart1,floatpart2=indv.variants
    #     binary_part1,binary_part2=binarize(floatpart1,eps=1,length=length_part1),binarize(floatpart2,eps=1,length=length_part2)
    #     x_decode=binary_part1+binary_part2
    #     return x_decode


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

    write_output(best_state, outputpath, filename)




