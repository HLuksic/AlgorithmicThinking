""" Algorithmic Thinking Project #2 """

from collections import deque

def bfs_visited(ugraph, start_node):
    """
    takes the undirected graph ugraph and the node start_node 
    and returns the set consisting of all nodes that are visited 
    by a breadth-first search that starts at start_node
    """
    queue = deque()
    visited = set([start_node])
    queue.append(start_node)
    while len(queue) > 0:
        node = queue.popleft()
        for neighbor in ugraph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return visited

def cc_visited(ugraph):
    """
    takes the undirected graph ugraph and returns a list of sets, 
    where each set consists of all the nodes (and nothing else) 
    in a connected component, and there is exactly one set in the 
    list for each connected component in ugraph and nothing else
    """
    remaining_nodes = set(ugraph.keys())
    connected_components = []
    while len(remaining_nodes) > 0:
        node = remaining_nodes.pop()
        visited = bfs_visited(ugraph, node)
        connected_components.append(visited)
        remaining_nodes.difference_update(visited)
    return connected_components

def largest_cc_size(ugraph):
    """
    takes the undirected graph ugraph and returns the size 
    (an integer) of the largest connected component in ugraph
    """
    connected_components = cc_visited(ugraph)
    max_size = 0
    for component in connected_components:
        if len(component) > max_size:
            max_size = len(component)
    return max_size

def compute_resilience(ugraph, attack_order):
    """
    takes the undirected graph ugraph, a list of nodes attack_order 
    and iterates through the nodes in attack_order. For each node 
    in the list, the function removes the given node and its edges 
    from the graph and then computes the size of the largest connected 
    component for the resulting graph. at [0] is the largest connected 
    component of the original graph and at [k+1] is is the size of the
    largest connected component after removing the first k nodes in attack_order
    """
    resilience = [largest_cc_size(ugraph)]
    for node in attack_order:
        ugraph.pop(node)
        for neighbor in ugraph:
            ugraph[neighbor].discard(node)
        resilience.append(largest_cc_size(ugraph))
    return resilience

if __name__ == "__main__":
    pass