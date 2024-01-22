""" Algorithmic Thinking Project #1 """

EX_GRAPH0 = {0: set([1, 2]), 
             1: set([]), 
             2: set([])}

EX_GRAPH1 = {0: set([1, 4, 5]),
             1: set([2, 6]),
             2: set([3]),
             3: set([0]),
             4: set([1]),
             5: set([2]),
             6: set([])}

EX_GRAPH2 = {0: set([1, 4, 5]),
             1: set([2, 6]),
             2: set([3, 7]),
             3: set([7]),
             4: set([1]),
             5: set([2]),
             6: set([]),
             7: set([3]),
             8: set([1, 2]),
             9: set([0, 3, 4, 5, 6, 7])}

def make_complete_graph(num_nodes):
    """ 
    returns a dictionary representation of a complete 
    graph with the specified number of nodes
    """
    graph = {}
    if num_nodes > 0:
        for node in range(num_nodes):
            graph[node] = set(range(num_nodes))
            graph[node].remove(node)
    elif num_nodes < 0:
        for node in range(-num_nodes):
            graph[node] = set()
    return graph

def compute_in_degrees(digraph):
    """ 
    computes the in-degrees for the nodes in the graph
    """
    in_degrees = {}
    for node in digraph:
        in_degrees[node] = 0
    for node in digraph:
        for edge in digraph[node]:
            in_degrees[edge] += 1
    return in_degrees

def in_degree_distribution(digraph):
    """
    computes the number of occurences of each in-degree
    """
    in_degree_dist = {}
    in_degrees = compute_in_degrees(digraph)
    for node in in_degrees:
        in_degree_dist[in_degrees[node]] = 0
    for node in in_degrees:
        in_degree_dist[in_degrees[node]] += 1        
    return in_degree_dist

if __name__ == "__main__":
    print(compute_in_degrees(EX_GRAPH1))
    print(in_degree_distribution(EX_GRAPH1))