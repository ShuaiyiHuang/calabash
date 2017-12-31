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


def build_graph(n, edges):
    graph = np.zeros((2*n+1, 2*n+1))
    get_node = lambda x: abs(x)+ (x>0)*n
    for i, edge in enumerate(edges):
        u, v, w = edge
        u, v = get_node(u), get_node(v)
        graph[u,v] = w
    return graph

def calc_power(states, graph):
    n = len(states)
    get_node = lambda x: abs(x)+ (x>0)*n
    states = [get_node(x) for x in states]
    states.insert(0,0)
    c_graph = graph[states]
    c_graph = c_graph[:,states]
    # import pdb; pdb.set_trace()
    mat_l = -c_graph # This operation will invoke copy operation
    ind = list(range(n+1))
    mat_l[ind, ind] = np.sum(c_graph, axis=0)
    det = np.linalg.det(mat_l[1:, 1:])
    return det

def power_by_mtt_fast2(state, graph):
    """Calculate the total power of the state, by the matrix-tree theorem with vectorized code.
    """
    n = len(state)
    get_node = lambda x: abs(x)+ (x>0)*n
    state = [get_node(x) for x in state]
    state.insert(0,0)
    s_graph = graph[np.array(state)]
    # import pdb;pdb.set_trace()
    s_graph = s_graph[:,state]

    diag=np.diagonal(s_graph)
    colum_sum_vec=np.sum(s_graph,0)
    new_digvalue=colum_sum_vec-diag
    reverse_graph=-s_graph

    indices_diag=np.diag_indices(n+1)

    mat_l=reverse_graph
    mat_l[indices_diag]=new_digvalue

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
    # input=int(np.around(input))
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



if __name__ == '__main__':
    input_path='./input/1'
    # n,edges=read_input(input_path)
    from calabash_week4_v1 import construct_graph,power_by_mtt_fastgraph

    n, edges, graph = construct_graph(input_path)
    # state = tuple(i * (-1) ** random.randrange(1, 3) for i in range(1, n + 1))
    # print('state:',state)
    # elap1=0.0
    # elap2=0.0
    # graph = build_graph(n, edges)
    # for i in range(6000):
    #     state = tuple(i * (-1) ** random.randrange(1, 3) for i in range(1, n + 1))
    #     startt=time.time()
    #     power_orig=power_by_mtt(state,edges)
    #     endt=time.time()
    #     elap1+=endt-startt
    #
    #     startt = time.time()
    #     # power_fast=power_by_mtt_fast2(state,graph)
    #     power_fast = calc_power(state,graph)
    #     endt=time.time()
    #     elap2+=endt-startt
    #     assert (power_orig - power_fast <1e-6)
    # print('elap_orig:',elap1,'elap_fast:',elap2,'fast',elap1-elap2)

    # state=decode_int(28108,n)
    # #test_state=(1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    # test_state=(0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0)
    # print('testis :',test_state)
    # print('mystate:',state)
    # assert (test_state==state)
    # print('state:', state)
    # power_fast = power_by_mtt_fast(state, edges)
    # print('best power',power_fast)

    #input 1
    length_part1 = int(np.floor(n * 0.5))
    maxValue_part1=2**length_part1

    length_part2=n-length_part1
    maxValue_part2=2**length_part2

    states=[-1 ,-2 ,+3 ,+4 ,+5 ,+6, -7 ,+8, -9, +10 ,+11 ,-12, -13 ,+14 ,-15, +16 ,-17, +18 ,-19, +20 ,-21 ,-22 ,-23, -24 ,+25 ,-26, +27, -28 ,-29 ,+30 ,-31 ,+32 ,-33 ,-34 ,+35, -36 ,+37 ,-38 ,+39 ,+40, +41 ,+42 ,-43 ,-44, +45 ,+46 ,+47 ,+48 ,+49 ,-50, +51 ,-52, -53 ,-54 ,-55 ,-56 ,-57, +58 ,-59 ,-60, +61 ,+62, -63 ,+64 ,-65 ,+66 ,-67 ,+68, -69]
    states_part1=states[0:length_part1]
    states_part2=states[length_part1:n]
    assert (len(states_part2)==length_part2)
    binary_str_part1=state_to_binary(states_part1)
    binary_str_part2=state_to_binary(states_part2)
    variant1=decimalize(binary_str_part1,eps=1,lower_bound=0)
    variant2=decimalize(binary_str_part2,eps=1,lower_bound=0)
    print('variants:',variant1,variant2)

    chrom=[1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1]
    chrom_part1=chrom[0:length_part1]
    chrom_part2=chrom[length_part1:n]
    chrom_v1=decimalize(chrom_part1,eps=1,lower_bound=0)
    chrom_v2=decimalize(chrom_part2,eps=1,lower_bound=0)
    print('from chrom variants:',variant1,variant2)

    best_power=power_by_mtt_fastgraph(states,graph)
    print('best power:',best_power)
