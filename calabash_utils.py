import numpy as np
import time
import random
from numpy import unravel_index
import os
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

def binarize(decimal, eps, length):
    '''
    Helper function to convert a float to binary sequence.
    :param decimal: the decimal number to be converted.
    :param eps: the decrete precision of binary sequence.
    :param length: the length of binary sequence.
    '''
    n = int(decimal/eps)
    bin_str = '{:0>{}b}'.format(n, length)
    return [int(i) for i in bin_str]
def decimalize(binary, eps, lower_bound):
    '''
    Helper function to convert a binary sequence back to decimal number.
    '''
    bin_str = ''.join([str(bit) for bit in binary])
    return lower_bound + int(bin_str, 2)*eps

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

def construct_greedy_graph(n,edges):
    import random
    random.seed(0)

    # n, edges = read_input(inputpath)
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
                # print('before 0:', neg_toneg[indu, indv])
                # print('v>0:', u, v, w)
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

    # print('preliminary negtoneg,negtopost,postoneg,postopos matrix:', matrix)
    return n,edges,matrix

def greedy_algorithm(n,edges):
    startt=time.time()

    n,edges,matrix=construct_greedy_graph(n,edges)
    mystate=[]
    nodeid_prev=0
    for i in range(1,n+1):
        # print('iter',i,'------------------------')
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
        # print('max indice:',indice,maxind_row,maxind_col)
        # print('maxweight:',maxw)

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
    print('greedy states:',state)

    #ordered
    state_ordered=order(state)
    return state_ordered

def greedy_algorithm2(n,edges):
    #not works well
    startt=time.time()

    n,edges,graph=construct_greedy_graph(n,edges)
    mystate=[]

    for i in range(1,n+1):
        print('iter',i,'------------------------')
        fromneg=[0,1]
        frompos=[2,3]
        graph_fromneg=graph[fromneg,i,:]
        assert (graph_fromneg.shape==(2,n+1))
        graph_frompos=graph[frompos,i,:]
        weight_fromneg=np.sum(graph_fromneg)
        weight_frompos=np.sum(graph_frompos)
        nodeid=i if weight_frompos>weight_fromneg else -i
        mystate.append(nodeid)
        print('weightfromneg,weightfrompos,chosenid:',weight_fromneg,weight_frompos,nodeid)

    state=tuple(mystate)

    endt=time.time()
    elapsed_time=endt-startt
    print ('greedy elapsed time:',elapsed_time//60,'min',elapsed_time%60,'s')

    return state

def order(states):
    #input list,unordered
    states_np=np.array(states)
    states_np_abs=abs(states_np)
    arg=np.argsort(states_np_abs)
    states_np_order=states_np[arg]
    states_list_order=list(states_np_order)
    return states_list_order

def state_to_binary(states):
    binary_str=[]
    for i,state in enumerate(states):
        if state>0:
            binary_str.append('0')
        elif state<0:
            binary_str.append('1')
    return binary_str

def decimalize(binary, eps, lower_bound):
    '''
    Helper function to convert a binary sequence back to decimal number.
    '''
    bin_str = ''.join([str(bit) for bit in binary])
    return lower_bound + int(bin_str, 2)*eps

def split_state_to_variants(states,length_part1,length_part2,n):
    states_part1=states[0:length_part1]
    states_part2=states[length_part1:n]
    assert (len(states_part2)==length_part2)
    binary_str_part1=state_to_binary(states_part1)
    binary_str_part2=state_to_binary(states_part2)
    variant1=decimalize(binary_str_part1,eps=1,lower_bound=0)
    variant2=decimalize(binary_str_part2,eps=1,lower_bound=0)

    return variant1,variant2

def greedy_initialize_variants(n,edges,length_part1,length_part2,graph):
    states=greedy_algorithm(n,edges)
    best_power = power_by_mtt_fastgraph(states, graph)
    print('greedy power',best_power)
    variant1,variant2=split_state_to_variants(states,length_part1,length_part2,n)
    return variant1,variant2

def random_plus(best_states,best_power,graph,flip_length_list=[1],times_list=[1500]):
    state_rand = list(best_states)
    select_list=range(1,n)
    # times=3000
    for idx,flip in enumerate(flip_length_list):
        times=times_list[idx]
        print('flip length:', flip, times)
        for i in range(times):
            rand_id=random.sample(select_list,flip)
            # print('rand_id',rand_id)
            new_state=list(state_rand)
            for ind in rand_id:
                new_state[ind]*=-1
            power = power_by_mtt_fastgraph(new_state, graph)
            if power>best_power:
                best_power=power
                print('update at node,',i+1,'new power:',best_power)
                # state=new_states
                best_states=list(new_state)
    print('greedy best power:',best_power)
    print('best states:',best_states)
    assert (best_power==power_by_mtt_fastgraph(best_states,graph))
    return best_power,best_states

def random_plus(best_states,best_power,graph,flip_length_list=[1],times_list=[1500]):
    state_rand = list(best_states)
    n=len(best_states)
    select_list=range(0,n)
    # times=3000
    for idx,flip in enumerate(flip_length_list):
        if idx==0:
            times=n
            print('idx', idx, 'times', times)
        elif idx==1:
            times=n
            print('idx', idx, 'times', times)
        elif idx==2:
            times=n
            print('idx', idx, 'times', times)
        else:
            times = times_list[idx]
        print('flip length:', flip, times)
        for i in range(times):
            if idx==0:
                rand_id=[select_list[i]]
                new_state=list(state_rand)
                for ind in rand_id:
                    new_state[ind]*=-1
                power = power_by_mtt_fastgraph(new_state, graph)
                if power>best_power:
                    best_power=power
                    print('update at node,',i+1,'new power:',best_power)
                    # state=new_states
                    best_states=list(new_state)
            elif idx==1:
                for j in range(times):
                    if j==i:
                        continue
                    rand_id=[select_list[i],select_list[j]]
                    new_state = list(state_rand)
                    for ind in rand_id:
                        new_state[ind] *= -1
                    power = power_by_mtt_fastgraph(new_state, graph)
                    if power > best_power:
                        best_power = power
                        print('update at node,', i + 1, 'new power:', best_power)
                        # state=new_states
                        best_states = list(new_state)
            elif idx==2:
                for j in range(times):
                    for k in range(times):
                        rand_id=[select_list[i],select_list[j],select_list[k]]
                        new_state = list(state_rand)
                        for ind in rand_id:
                            new_state[ind] *= -1
                        power = power_by_mtt_fastgraph(new_state, graph)
                        if power > best_power:
                            best_power = power
                            print('update at node,', i + 1, 'new power:', best_power)
                            # state=new_states
                            best_states = list(new_state)
            else:
                rand_id=random.sample(select_list,flip)
            # print('rand_id',rand_id)
                new_state=list(state_rand)
                for ind in rand_id:
                    new_state[ind]*=-1
                power = power_by_mtt_fastgraph(new_state, graph)
                if power>best_power:
                    best_power=power
                    print('update at node,',i+1,'new power:',best_power)
                    # state=new_states
                    best_states=list(new_state)
    print('greedy best power:',best_power)
    print('best states:',best_states)
    assert (best_power==power_by_mtt_fastgraph(best_states,graph))
    return best_power,best_states

def power_to_score(power):
    score=int(1e9 + 1e6*np.log(power))
    return score

if __name__ == '__main__':
    print()
    # import parser
    # parser.add_argument('--rseed', type=int, default=30,
    #                     help='random.seed')
    # parser.add_argument('--npseed', type=int, default=32,
    #                     help='np.random.seed')
    startt=time.time()

    inputdpath='./input'
    outputpath='./output'
    filename='6'
    inputdir = os.path.join(inputdpath, filename)
    import argparse
    parser = argparse.ArgumentParser(description='tune calabash')

    parser.add_argument('--rseed', type=int, default=325986,
                        help='random.seed')
    parser.add_argument('--npseed', type=int, default=12378,
                        help='np.random.seed')
    args = parser.parse_args()
    rseed=args.rseed
    npseed=args.npseed
    random.seed(rseed)
    np.random.seed(npseed)

    print(args)
    n, edges, graph = construct_graph(inputdir)
    # state=greedy_algorithm(n,edges)
    # best_states=order(state)
    # best_power=power_by_mtt_fastgraph(best_states,graph)
    # print('greedy power:',best_power)

    # a='-1 -2 -3 -4 -5 -6 7 -8 -9 -10 11 12 -13 14 -15 -16 -17 18 19 20 21 -22 -23 -24 25 -26 27 28 29 -30 -31 -32 -33 -34 35 -36 -37 38 39 40 -41 -42 43 -44 45 -46 47 48 -49 50 51 52 -53 -54 -55 56 57 -58 59 -60 61 62 63 -64 -65 -66 67 68 69 70 -71 72 -73 74 -75 76 -77 78 -79 80 81 82 83 -84 85 -86 87 -88 -89 90 91 -92 93 -94 -95 -96 -97 -98 -99 -100'
    # bin_str=a.split(' ')
    # best_states = [int(str_bit) for str_bit in bin_str]
    best_states=[-1, 2, -3, 4, -5, 6, -7, 8, 9, 10, 11, 12, 13, -14, -15, -16, 17, 18, 19, 20, -21, 22, -23, 24, -25, -26, 27, -28, -29, 30, -31, -32, 33, 34, -35, -36, 37, -38, 39, -40, 41, -42, -43, 44, -45, 46, -47, -48, 49, -50, -51, -52, -53, 54, -55, 56, -57, 58, -59, 60, 61, 62, 63, -64, -65, 66, -67, 68, -69, 70, 71, 72, 73, -74, 75, -76, 77, 78, -79, 80, -81, -82, -83, 84, -85, 86, -87, 88, 89, -90, 91, -92, -93, 94, 95, 96, 97, 98, -99, -100]
    best_power=power_by_mtt_fastgraph(best_states,graph)
    print('initial power:', best_power)
    score=power_to_score(best_power)
    print('initial score',score)

    flip_length_list=[1,2,3,4,5,6]
    times_list=[5*n,n*n,int(n/2)*n*n,int(n/2)*n*n,int(n/2)*n*n,int(n/2)*n*n]
    best_power, best_states=random_plus(best_states,best_power,graph,flip_length_list,times_list)

    #score
    print('new power',best_power)
    score=int(1e9 + 1e6*np.log(best_power))
    print('score:',score)
    write_output(best_states,outputpath,filename)

    endt=time.time()
    print('cost time',(endt-startt)%60,'min',(endt-startt)/60,'s')

