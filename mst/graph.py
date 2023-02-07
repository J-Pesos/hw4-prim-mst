import numpy as np
import heapq as hq
import random
from typing import Union

class Graph:

    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """
    
        Unlike the BFS assignment, this Graph class takes an adjacency matrix as input. `adjacency_mat` 
        can either be a 2D numpy array of floats or a path to a CSV file containing a 2D numpy array of floats.

        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph.
    
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else: 
            raise TypeError('Input must be a valid path or an adjacency matrix')
        self.mst = None

    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=',')

    def construct_mst(self):
        """
    
        TODO: Given `self.adj_mat`, the adjacency matrix of a connected undirected graph, implement Prim's 
        algorithm to construct an adjacency matrix encoding the minimum spanning tree of `self.adj_mat`. 
            
        `self.adj_mat` is a 2D numpy array of floats. Note that because we assume our input graph is
        undirected, `self.adj_mat` is symmetric. Row i and column j represents the edge weight between
        vertex i and vertex j. An edge weight of zero indicates that no edge exists. 
        
        This function does not return anything. Instead, store the adjacency matrix representation
        of the minimum spanning tree of `self.adj_mat` in `self.mst`. We highly encourage the
        use of priority queues in your implementation. Refer to the heapq module, particularly the 
        `heapify`, `heappop`, and `heappush` functions.

        """
        
        # Initialize minimum-spanning tree array of same size as adjacency matrix array.
        MST = np.zeros(self.adj_mat.shape)
        # Initialize number of vertices that are in the network.
        vertices = np.shape(self.adj_mat)[0]
        # Select a random start vertex in network.
        start = random.choice(range(0, vertices))
        # Initialize priority queue.
        priority_queue = []
        # Initialize list of visited vertices.
        visited = []

        # For all other vertices in the network find edges connecting to start vertex, then add weight, starting, and end vertex to priority queue.
        for vertex in range(vertices): # Loop through all vertex indices.
            if self.adj_mat[start, vertex] != 0: # If edge exists.
                priority_queue.append((self.adj_mat[start, vertex], start, vertex)) # Append edge and edge weight to priority queue.

        # Append visited vertex.
        visited.append(start)
        # Turn priority queue into heap queue.
        hq.heapify(priority_queue)

        while len(visited) != vertices: # While length of visited vertices is less than total amount.
            edge_weight, start, end = hq.heappop(priority_queue) # Pop and return smallest edge and edge weight from heap queue, maintaing heap invariant.
            if end not in visited: # If end vertex has not yet been visited.
                MST[start, end] = edge_weight # Add edge and edge weight to MST.
                MST[end, start] = edge_weight
                visited.append(end) # Add end vertex to visited.
                
                for vertex in range(vertices): # Add new destinations from end vertex into priority queue.
                    if self.adj_mat[end, vertex] != 0:
                        hq.heappush(priority_queue, (self.adj_mat[end, vertex], end, vertex))
                        
        self.mst = MST
