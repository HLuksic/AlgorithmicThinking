""" Algorithmic Thinking Application #1 """

"""
Provided code for Application portion of Module 1

Imports physics citation graph 
"""

# general imports
import urllib.request as req
import matplotlib.pyplot as plt
import random

# Set timeout for CodeSkulptor if necessary
#import codeskulptor
#codeskulptor.set_timeout(20)


###################################
# Code for loading citation graph

CITATION_URL = "http://storage.googleapis.com/codeskulptor-alg/alg_phys-cite.txt"

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

def normalized_distribution(digraph):
    """
    normalizes the distribution
    """
    dist = in_degree_distribution(digraph)
    total = sum(dist.values())
    for key in dist:
        dist[key] /= total
    return dist

def plot_distribution(digraph):
    """
    plots the distribution
    """
    dist = normalized_distribution(digraph)
    degrees = list(dist.keys())
    values = list(dist.values())
    plt.loglog(degrees, values, 'ro')
    plt.xlabel('in-degree')
    plt.ylabel('normalized # of papers')
    plt.title('In-degree distribution of DPA graph')
    plt.savefig("distributionDPA.png")
    print("Done!")

def random_digraph_er(num_nodes, prob):
    """
    generates a random directed graph
    """
    digraph = {}
    for node in range(num_nodes):
        digraph[node] = set([])
    for node in range(num_nodes):
        for edge in range(num_nodes):
            if node != edge:
                a = random.random()
                if a < prob:
                    digraph[node].add(edge)
    return digraph

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

def random_digraph_dpa(num_nodes, num_subset):
    """
    generates a random directed graph using DPA algorithm
    """
    digraph = make_complete_graph(num_subset)
    in_degrees = compute_in_degrees(digraph)
    totalindeg = sum(in_degrees.values())
    for node_num in range(num_subset, num_nodes):
        subset_digraph = {}
        while len(subset_digraph) < num_subset:
            node = random.choice(range(len(digraph)))
            a = random.random()
            if a < (in_degrees[node] + 1) / (totalindeg + len(digraph)) * 70:
                subset_digraph[node] = digraph[node]
        if node_num % 100 == 0: 
            print(node_num)
        # add the new node to digraph
        digraph[node_num] = set([])
        in_degrees[node_num] = 0
        # add edges to the new node
        for node in subset_digraph:
            digraph[node_num].add(node)
            in_degrees[node] += 1
            totalindeg += 1
    return digraph

if __name__ == "__main__":
    # graph = load_graph(CITATION_URL)
    # graph = random_digraph_er(1000, 0.5)
    graph = random_digraph_dpa(27700, 13)
    plot_distribution(graph)
