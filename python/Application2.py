""" Algorithmic Thinking Application #1 """

"""
Provided code for application portion of module 2

Helper class for implementing efficient version
of UPA algorithm
"""

import random

class UPATrial:
    """
    Simple class to encapsulate optimizated trials for the UPA algorithm
    
    Maintains a list of node numbers with multiple instance of each number.
    The number of instances of each node number are
    in the same proportion as the desired probabilities
    
    Uses random.choice() to select a node number from this list for each trial.
    """

    def __init__(self, num_nodes):
        """
        Initialize a UPATrial object corresponding to a 
        complete graph with num_nodes nodes
        
        Note the initial list of node numbers has num_nodes copies of
        each node number
        """
        self._num_nodes = num_nodes
        self._node_numbers = [node for node in range(num_nodes) for dummy_idx in range(num_nodes)]


    def run_trial(self, num_nodes):
        """
        Conduct num_nodes trials using by applying random.choice()
        to the list of node numbers
        
        Updates the list of node numbers so that each node number
        appears in correct ratio
        
        Returns:
        Set of nodes
        """
        
        # compute the neighbors for the newly-created node
        new_node_neighbors = set()
        for _ in range(num_nodes):
            new_node_neighbors.add(random.choice(self._node_numbers))
        
        # update the list of node numbers so that each node number 
        # appears in the correct ratio
        self._node_numbers.append(self._num_nodes)
        for dummy_idx in range(len(new_node_neighbors)):
            self._node_numbers.append(self._num_nodes)
        self._node_numbers.extend(list(new_node_neighbors))
        
        #update the number of nodes
        self._num_nodes += 1
        return new_node_neighbors



"""
Provided code for Application portion of Module 2
"""

# general imports
from matplotlib import pyplot as plt
from collections import deque
import urllib.request as req
import random
import time
import math

# CodeSkulptor import
#import simpleplot
#import codeskulptor
#codeskulptor.set_timeout(60)

# Desktop imports
#import matplotlib.pyplot as plt


############################################
# Provided code

def copy_graph(graph):
    """
    Make a copy of a graph
    """
    new_graph = {}
    for node in graph:
        new_graph[node] = set(graph[node])
    return new_graph

def delete_node(ugraph, node):
    """
    Delete a node from an undirected graph
    """
    neighbors = ugraph[node]
    ugraph.pop(node)
    for neighbor in neighbors:
        ugraph[neighbor].remove(node)
    
def targeted_order(ugraph):
    """
    Compute a targeted attack order consisting
    of nodes of maximal degree
    
    Returns:
    A list of nodes
    """
    # copy the graph
    new_graph = copy_graph(ugraph)
    
    order = []    
    while len(new_graph) > 0:
        max_degree = -1
        for node in new_graph:
            if len(new_graph[node]) > max_degree:
                max_degree = len(new_graph[node])
                max_degree_node = node
        
        neighbors = new_graph[max_degree_node]
        new_graph.pop(max_degree_node)
        for neighbor in neighbors:
            new_graph[neighbor].remove(max_degree_node)

        order.append(max_degree_node)
    return order


##########################################################
# Code for loading computer network graph

NETWORK_URL = "http://storage.googleapis.com/codeskulptor-alg/alg_rf7.txt"


def load_graph(graph_url):
    """
    Function that loads a graph given the URL
    for a text representation of the graph
    
    Returns a dictionary that models a graph
    """
    graph_file = req.urlopen(graph_url)
    graph_text = graph_file.read()
    graph_lines = graph_text.split(b'\n')
    graph_lines = graph_lines[ : -1]
    
    print(f"Loaded graph with {len(graph_lines)} nodes")
    
    answer_graph = {}
    for line in graph_lines:
        neighbors = line.split(b' ')
        node = int(neighbors[0])
        answer_graph[node] = set([])
        for neighbor in neighbors[1 : -1]:
            answer_graph[node].add(int(neighbor))

    return answer_graph


def random_ugraph_er(num_nodes, prob):
    """
    generates a random unidrected graph
    """
    graph = {}
    for node in range(num_nodes):
        graph[node] = set([])
    for node1 in range(num_nodes):
        for node2 in range(num_nodes):
            if node1 != node2:
                a = random.random()
                if a < prob:
                    graph[node1].add(node2)
                    graph[node2].add(node1)
    return graph


def make_complete_ugraph(num_nodes):
    """ 
    returns a dictionary representation of a complete 
    undirected graph with the specified number of nodes
    """
    graph = {}
    for node in range(num_nodes):
        graph[node] = set([])
    for node1 in range(num_nodes):
        for node2 in range(num_nodes):
            if node1 != node2:
                graph[node1].add(node2)
                graph[node2].add(node1)
    return graph


def random_ugraph_upa(n, m):
    graph_dict = make_complete_ugraph(m)
    graph = UPATrial(m)

    for i in range(m, n):
        neighbors = graph.run_trial(m)
        graph_dict[i] = neighbors
        for neighbor in neighbors:
            graph_dict[neighbor].add(i)

    return graph_dict


def count_edges(graph):
    """
    counts the number of edges in an undirected graph
    """
    count = 0
    for node in graph:
        count += len(graph[node])
    return count // 2


def random_order(graph):
    """
    takes a graph and returns a list of the nodes in the graph in some random order
    """
    nodes = list(graph.keys())
    random.shuffle(nodes)
    return nodes


def bfs_visited(ugraph, start_node):
    """
    function from Project2.py
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
    function from Project2.py
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
    function from Project2.py
    """
    connected_components = cc_visited(ugraph)
    max_size = 0
    for component in connected_components:
        if len(component) > max_size:
            max_size = len(component)
    return max_size

def compute_resilience(ugraph, attack_order):
    """
    function from Project2.py
    """
    resilience = [largest_cc_size(ugraph)]
    for node in attack_order:
        ugraph.pop(node)
        for neighbor in ugraph:
            ugraph[neighbor].discard(node)
        resilience.append(largest_cc_size(ugraph))
    return resilience


def fast_targeted_order(ugraph):
    """
    return graph nodes in decreasing order of their degrees
    """
    graph = copy_graph(ugraph)
    node_list = []
    deg_sets = {}

    for i in range(0, len(graph)):
        deg_sets[i] = set()

    for key, value in graph.items():
        deg = len(value)
        deg_sets[deg].add(key)

    for i in range((len(graph) - 1), -1, -1):
        while len(deg_sets[i]) > 0:
            each = deg_sets[i].pop()
            if len(graph[each]):
                for neighbor in graph[each]:
                    if neighbor in graph:
                        d = len(graph[neighbor])
                        if neighbor in deg_sets[d]:
                            deg_sets[d].remove(neighbor)
                            deg_sets[d-1].add(neighbor)
            node_list.append(each)
            del graph[each]

    return node_list


def plot_resilience(order):
    graph_online = load_graph(NETWORK_URL)
    p = 0.002
    m = 3
    print(f"p = {p}, m = {m}")
    graph_er = random_ugraph_er(1239, p)
    graph_upa = random_ugraph_upa(1239, m)
    print(f"Online graph edges: {count_edges(graph_online)}")
    print(f"Random ER graph edges: {count_edges(graph_er)}")
    print(f"Random UPA graph edges: {count_edges(graph_upa)}")
    # plot the results as three curves combined in a single standard plot (not log/log). Use a line plot for each curve. The horizontal axis for your single plot be the the number of nodes removed (ranging from zero to the number of nodes in the graph) while the vertical axis should be the size of the largest connect component in the graphs resulting from the node removal
    x = list(range(1240))
    y1 = compute_resilience(graph_online, order(graph_online))
    y2 = compute_resilience(graph_er, order(graph_er))
    y3 = compute_resilience(graph_upa, order(graph_upa))
    plt.plot(x, y1, '-b', label='Online graph')
    plt.plot(x, y2, '-r', label='ER graph')
    plt.plot(x, y3, '-g', label='UPA graph')
    plt.legend(loc='upper right')
    plt.xlabel('Number of nodes removed')
    plt.ylabel('Size of the largest connected component')
    plt.title('Resilience of three graphs (targeted attack order)')
    plt.savefig('resilience_targeted.png')


def plot_performance():
    """
    plot the running times of targeted_order and fast_targeted_order
    """
    x = []
    y1 = []
    y2 = []
    for i in range(10, 1000, 10):
        x.append(i)
        graph = random_ugraph_upa(i, 5)
        start = time.time()
        targeted_order(graph)
        end = time.time()
        y1.append(end - start)
        start = time.time()
        fast_targeted_order(graph)
        end = time.time()
        y2.append(end - start)
    plt.plot(x, y1, '-b', label='Targeted order O(n^2)')
    plt.plot(x, y2, '-r', label='Fast targeted order O(n)')
    plt.legend(loc='upper right')
    plt.xlabel('Number of nodes')
    plt.ylabel('Running time')
    plt.title('Running time of given and optimized attack order algorithm')
    plt.savefig('performance.png')


if __name__ == "__main__":
    #plot_resilience(fast_targeted_order)
    plot_performance()