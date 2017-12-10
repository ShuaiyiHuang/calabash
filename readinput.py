import os
import numpy as np
import random
import time
def read_input(input_path):
    if not os.path.exists(input_path):
        print('Error:input file not exitst!')
        return None,None

    edges=[]
    with open(input_path) as f:
        lines = f.read().splitlines()
        line0=lines[0]
        print('line0:',line0)
        n=int(line0)
        for i,line in enumerate(lines[1:]):
            u, v, w = line.split()
            one_edge=(int(u), int(v), float(w))
            edges.append(one_edge)
        f.close()
    return n,edges

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
    # print('graph:',graph)
    # print('mat_l',mat_l)
    print('orig det',det)
    return det

def power_by_mtt_fast_debug(state,edges):
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
    # print('diag:',diag)
    # print('colum_sum_vec:',colum_sum_vec)
    # print('new_diagvalue:',new_digvalue)
    reverse_graph=-graph
    # print('reverse_graph',reverse_graph)
    indices_diag=np.diag_indices(n+1)
    # print('indices_diag:',indices_diag)
    mat_l=reverse_graph
    mat_l[indices_diag]=new_digvalue
    # print('mat_l:',mat_l)
    det = np.linalg.det(mat_l[1:, 1:])
    # print('new det:',det)

    return det

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
    print('new det:',det)
    return det

def decode_int(input,numNode):
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

if __name__ == '__main__':
    input_path='./input/1'
    n,edges=read_input(input_path)
    # state = tuple(i * (-1) ** random.randrange(1, 3) for i in range(1, n + 1))
    # print('state:',state)
    # elap1=0.0
    # elap2=0.0
    # for i in range(6000):
    #     state = tuple(i * (-1) ** random.randrange(1, 3) for i in range(1, n + 1))
    #     startt=time.time()
    #     power_orig=power_by_mtt(state,edges)
    #     endt=time.time()
    #     elap1+=endt-startt
    #     startt = time.time()
    #     power_fast=power_by_mtt_fast(state,edges)
    #     endt=time.time()
    #     elap2+=endt-startt
    #     assert (power_orig == power_fast)
    # print('elap_orig:',elap1,'elap_fast:',elap2,'fast',elap1-elap2)

    state=decode_int(6,n)
    print('state:', state)
    power_fast = power_by_mtt_fast(state, edges)
    print('best power',power_fast)