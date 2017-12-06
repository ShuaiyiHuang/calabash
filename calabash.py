import numpy as np
import time
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
    import random
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
    print power

def greedy_algorithm():
    import random
    random.seed(0)

    n, edges = read_input()
    times = 1
    best_state, best_power = None, None

    startt=time.time()
    #for each starting store max weight
    pos_topos,pos_toneg,neg_topos,neg_toneg=4*[-1*np.ones((n+1,n+1),dtype=float)]

    for edge in edges:
        u,v,w=edge
        indu=abs(u)
        indv=abs(v)
        assert (type(u)==int and type(v)==int and type(w)==float)
        if u>0 and v>0:
            u, v = abs(u), abs(v)
            pos_topos[u,v]=w
        if u>0 and v<0:
            u, v = abs(u), abs(v)
            pos_toneg[u,v]=w
        if u<0 and v>0:
            u, v = abs(u), abs(v)
            neg_topos[u,v]=w
        if u<0 and v<0:
            print('neg to nge:',u,v)
            u, v = abs(u), abs(v)
            neg_toneg[u,v]=w
        if u==0:
            if v>0:
                print('before 0:',neg_toneg[indu,indv])
                print('v>0:',u,v,w)
                neg_topos[u,v]=w
                pos_topos[u,v]=w
                print('after 0:',neg_toneg[indu,indv])
            if v<0:
                print('v<0:',u,v,w)
                neg_toneg[u,abs(v)]=w
                pos_toneg[u,abs(v)]=w
    endt=time.time()
    assert (neg_topos[0,:].all()==pos_topos[0,:].all())
    assert (pos_toneg[0,:].all()==neg_toneg[0,:].all())
    print('index finished in {}'.format((endt-startt)))

    print ('neg to neg\n',neg_toneg)
    print('neg to pos:\n', neg_topos)
    print('pos to neg:/n', pos_toneg)
    print('pos to pos:' + '/n', pos_topos)

    mystate=[]

    matrix=np.concatenate([np.expand_dims(neg_toneg,0),np.expand_dims(neg_topos,0),np.expand_dims(pos_toneg,0),np.expand_dims(pos_topos,0)],0)
    print(matrix[0].shape)
    print(matrix[1].shape)
    print(matrix[3])

    nodeid_prev=0

    print('preliminary matrix:',matrix)
    for i in range(1,n+1):
        nodeid=-99

        fromneg=[0,1]
        frompos=[2,3]
        fromsign=[]
        if nodeid_prev<0:
            fromsign=fromneg
        else:
            fromsign=frompos

        vec = matrix[fromsign, nodeid_prev, :]
        print('vec shape:',vec.shape)
        print('vec:',vec)

        maxw=np.max(vec)
        indice=unravel_index(vec.argmax(), vec.shape)
        sign, nodeindex = indice
        assert (len(indice) == 2)
        print('max indice:',sign,nodeindex)
        print('maxw:',maxw)

        if(sign==0):
            #newly find nodeid is neg
            nodeid=-(abs(nodeindex))
        elif(sign==1):
            #newly find nodeid is pos
            nodeid=abs(nodeindex)

        #disable chosen id,set to negative
        print('before set to neg:',matrix[:,:,nodeindex])
        matrix[:,:,nodeindex]=-1
        matrix[:, nodeid_prev, :] = -1
        print('after set to neg:',matrix[:,:,nodeindex])
        print('noid:',nodeid,'matrix:',matrix)
        mystate.append(nodeid)
        nodeid_prev=nodeid
    print('mystate:',mystate)
    state=tuple(mystate)

    for _ in range(times):
        # state = tuple(i * (-1)**random.randrange(1, 3) for i in range(1, n+1))
        power = power_by_mtt(state, edges)
        if best_state is None or best_power < power:
            best_state = state
            best_power = power
    assert best_state is not None
    print ' '.join('%+d' % i for i in best_state)
    print power


def read_input():
    def one_edge():
        line = raw_input()
        u, v, w = line.split()
        return int(u), int(v), float(w)
    n = int(raw_input())
    edges = [one_edge() for _ in range(4 * n**2 - 2*n)]
    return n, edges


if __name__ == '__main__':
    #randomized_algorithm()
    greedy_algorithm()