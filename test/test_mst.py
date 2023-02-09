import pytest
import numpy as np
from mst import Graph
from sklearn.metrics import pairwise_distances
import networkx as nx


def check_mst(adj_mat: np.ndarray, 
              mst: np.ndarray, 
              expected_weight: int, 
              allowed_error: float = 0.0001):
    """
    
    Helper function to check the correctness of the adjacency matrix encoding an MST.
    Note that because the MST of a graph is not guaranteed to be unique, we cannot 
    simply check for equality against a known MST of a graph. 

    Arguments:
        adj_mat: adjacency matrix of full graph
        mst: adjacency matrix of proposed minimum spanning tree
        expected_weight: weight of the minimum spanning tree of the full graph
        allowed_error: allowed difference between proposed MST weight and `expected_weight`

    TODO: Add additional assertions to ensure the correctness of your MST implementation. For
    example, how many edges should a minimum spanning tree have? Are minimum spanning trees
    always connected? What else can you think of?

    """

    def approx_equal(a, b):
        return abs(a - b) < allowed_error

    total = 0
    for i in range(mst.shape[0]):
        for j in range(i+1):
            total += mst[i, j]
    assert approx_equal(total, expected_weight), 'Proposed MST has incorrect expected weight'

    # Ensure combined edge weights of mst are less than those of original graph.
    assert np.sum(adj_mat) > np.sum(mst), 'Weight of mst is not less than that of original graph.'

    # Ensure the minimum spanning tree is actually connected.
    mst_nx = nx.from_numpy_array(mst)
    assert nx.is_connected(mst_nx) == True, "Not all vertices of minimum spanning tree are connected."


def test_mst_small():
    """
    
    Unit test for the construction of a minimum spanning tree on a small graph.
    
    """
    file_path = './data/small.csv'
    g = Graph(file_path)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 8)


def test_mst_single_cell_data():
    """
    
    Unit test for the construction of a minimum spanning tree using single cell
    data, taken from the Slingshot R package.

    https://bioconductor.org/packages/release/bioc/html/slingshot.html

    """
    file_path = './data/slingshot_example.txt'
    coords = np.loadtxt(file_path) # load coordinates of single cells in low-dimensional subspace
    dist_mat = pairwise_distances(coords) # compute pairwise distances to form graph
    g = Graph(dist_mat)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 57.263561605571695)


def test_mst_student(allowed_error: float = 0.000001):
    """
    
    TODO: Write at least one unit test for MST construction.
    
    """
    # Ensure that minimum spanning trees for both small dataset and slingshot example are symmetric.
    small_path = './data/small.csv'
    slingshot_path = './data/slingshot_example.txt'

    # Create mst for small dataset.
    small_g = Graph(small_path)
    small_g.construct_mst()

    # Load coordinates of single-cell data and calculate pairwise distances to form numpy object.
    slingshot_coords = np.loadtxt(slingshot_path)
    slingshot_distances = pairwise_distances(slingshot_coords)
    
    # Create mst for single-cell data.
    slingshot_g = Graph(slingshot_distances)
    slingshot_g.construct_mst()

    # Check that both MSTs are symmetric.
    assert np.all(np.abs(small_g.mst - small_g.mst.T) < allowed_error) == True, "The mst for the small dataset is not symmetric."
    assert np.all(np.abs(slingshot_g.mst - slingshot_g.mst.T) < allowed_error) == True, "The mst for the single-cell dataset is not symmetric."
