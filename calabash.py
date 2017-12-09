import numpy as np
import time
import random
from numpy import unravel_index

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
    print ' '.join('%+d' % i for i in best_state)
    print 'random algorithm best power:',best_power
    return best_power

def construct_graph():
    import random
    random.seed(0)

    n, edges = read_input()
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

    # print ('neg to neg\n', neg_toneg)
    # print('neg to pos:\n', neg_topos)
    # print('pos to neg:/n', pos_toneg)
    # print('pos to pos:' + '/n', pos_topos)

    state = np.arange(1,n+1)

    matrix = np.concatenate([np.expand_dims(neg_toneg, 0), np.expand_dims(neg_topos, 0), np.expand_dims(pos_toneg, 0),
                             np.expand_dims(pos_topos, 0)], 0)

    # print('preliminary negtoneg,negtopost,postoneg,postopos matrix:', matrix)
    return n,edges,matrix

def greedy_algorithm():
    startt=time.time()

    n,edges,matrix=construct_graph()
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

        #disable chosen id,set to negative
        # print('before set to neg:',matrix[:,:,maxind_col])
        matrix[:,:,maxind_col]=-1
        # print('after set to neg:',matrix[:,:,maxind_col])
        # print('chosen node state:',nodeid)
        # print('updated matrix:',matrix)
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
    print ' '.join('%+d' % i for i in best_state)
    print best_power
    return  best_power



def greedy_main():
    state,edges=greedy_algorithm()
    n=len(state)
    state=list(state)
    best_power=power_by_mtt(state,edges)
    print('initial states:',state)
    print('initial best power:',best_power)
    for i in range(n):
        # new_states = list(state)
        state[i]*=-1
        # print(i+1,'new state for comput:',new_states)
        power=power_by_mtt(state,edges)
        if power>best_power:
            best_power=power
            print('update at node,',i+1,'new power:',best_power)
            # state=new_states
            best_states=state

    print('greedy best power:',best_power)
    print('best states:',best_states)
    return best_power



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


def read_input():
    def one_edge():
        line = raw_input()
        u, v, w = line.split()
        return int(u), int(v), float(w)
    n = int(raw_input())
    edges = [one_edge() for _ in range(4 * n**2 - 2*n)]
    return n, edges


if __name__ == '__main__':
    # #
    # greedy_algorithm()
    # construct_graph()
    startt=time.time()

    greedy_main()
    # randomized_algorithm()
    # main()

    endtt=time.time()
    elapsed=endtt-startt
    print('runtime in total is ',elapsed//60,'min',elapsed%60,'s')