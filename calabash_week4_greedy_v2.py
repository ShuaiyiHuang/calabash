import numpy as np
import time
import random
from numpy import unravel_index
import os


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


def randomized_algorithm():

    random.seed(0)

    n, edges = read_input()
    times = 10
    best_state, best_power = None, None
    for _ in range(times):
        rand_state = tuple(i * (-1)**random.randrange(1, 3) for i in range(1, n+1))
        power = power_by_mtt(rand_state, edges)
        if best_state is None or best_power < power:
            best_state = rand_state
            best_power = power
    assert best_state is not None
    print(' '.join('%+d' % i for i in best_state))
    print('random algorithm best power:',best_power)
    return best_power

def construct_greedy_graph(inputpath):
    import random
    random.seed(0)

    n, edges = read_input(inputpath)
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

def greedy_algorithm(inputpath):
    startt=time.time()

    n,edges,matrix=construct_greedy_graph(inputpath)
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

    return state,edges

def compute_power(state,edges):
    times=1
    best_state,best_power=None,None
    for _ in range(times):
        # state = tuple(i * (-1)**random.randrange(1, 3) for i in range(1, n+1))
        power = power_by_mtt(state, edges)
        if best_state is None or best_power < power:
            best_state = state
            best_power = power
    assert best_state is not None
    print(' '.join('%+d' % i for i in best_state))
    print (best_power)
    return  best_power

def greedy_main(inputpath,graph):
    state,edges=greedy_algorithm(inputpath)
    n=len(state)
    state=list(state)
    print('initial states:',state)

    #initialize best
    best_power=power_by_mtt_fastgraph(state,graph)
    best_states=list(state)

    print('initial best power:',best_power)
    for i in range(n):
        # new_states = list(state)
        state[i]*=-1
        # print(i+1,'new state for comput:',new_states)
        power=power_by_mtt_fastgraph(state,graph)
        if power>best_power:
            best_power=power
            print('update at node,',i+1,'new power:',best_power)
            # state=new_states
            best_states=list(state)
    times=10
    print('random.......')
    state_rand = list(best_states)
    select_list=range(1,n+1)
    for i in range(times):
        rand_id=random.sample(select_list,2)
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



def main():
    power_rand,power_greed,best_power=None,None,None
    power_rand=randomized_algorithm()
    power_greed=greedy_main()
    print('greedy power:',power_greed)
    print('random power:',power_rand)
    if power_rand>power_greed:
        best_power=power_rand
        print('random algorithm is better,best power is')
    else:
        best_power=power_greed
        print('greedy algorithm is better,best power is')
    print(best_power)


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

def construct_whole_graph(input_path):
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

    inputdpath='./input'
    outputpath='./output'
    filename='4'
    inputdir = os.path.join(inputdpath, filename)

    startt = time.time()
    n, edges, graph = construct_whole_graph(inputdir)


    best_power,best_states=greedy_main(inputdir,graph)
    # best_states,edges=greedy_algorithm(inputdir)

    begin=time.time()
    best_power=power_by_mtt_fastgraph(best_states,graph)
    stop=time.time()
    print('best power:',best_power,'power cost time',stop-begin,'s')


    write_output(best_states,outputpath,filename)


    endtt=time.time()
    elapsed=endtt-startt
    print('runtime in total is ',elapsed//60,'min',elapsed%60,'s')