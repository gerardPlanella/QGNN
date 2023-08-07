import numpy as np
import torch
from dataset.qm9 import QM9Properties

def get_mean_and_mad(train_dataset, property):
    
    if isinstance(property, QM9Properties):
        values = []
        n = 0
        for batch in train_dataset:
            values.extend(batch.y[:, property.value])
            
        mean = torch.mean(torch.tensor(values))
        mad = torch.mean(torch.abs(torch.tensor(values) - mean))
        return mean, mad
    else:
        raise NotImplementedError

def lap_positional_encoding(graph, pos_encoding_dim=10):
    """
    Computes the Laplacian positional encoding of a graph.

    Args:
        graph (dict): A dictionary containing the graph information with 'edge_index' as the key.
                      The 'edge_index' should be a 2xM array where M is the number of edges.
        pos_encoding_dim (int, optional): The number of dimensions to encode the graph. Defaults to 10.

    Returns:
        torch.Tensor: A 2D tensor of shape (num_nodes, pos_encoding_dim) containing the (initial) Laplacian positional
                      encoding of the graph.
    """
    # Calculate the laplacian matrix
    edge_index = graph["edge_index"].numpy()
    num_nodes = np.max(edge_index) + 1
    adj = np.zeros((num_nodes, num_nodes))
    adj[edge_index[0], edge_index[1]] = 1
    adj[edge_index[1], edge_index[0]] = 1
    deg_n = np.sum(adj, axis=0) ** -0.5

    # Compute laplacian via factorization of graph laplacian
    lap = np.eye(adj.shape[0]) - deg_n * adj * deg_n

    # Get Eigenvectors
    EigVal, EigVec = np.linalg.eig(lap)
    idx = EigVal.argsort()  # sort increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])

    # Return the eigen vectors up to the set encoding dimension
    return torch.from_numpy(EigVec[:, 1:pos_encoding_dim+1],).float()

def rw_positional_encoding(graph, depth=10):
    """
    Conducts a random walk on a given graph and returns the random walk matrix and Laplacian matrix.

    Args:
        graph (dict): A dictionary containing the graph information with 'edge_index' as the key.
                      The 'edge_index' should be a 2xM array where M is the number of edges.
        depth (int, optional): The maximum depth of the random walk. Defaults to 10.

    Returns:
        tuple: A tuple containing the following:
            - torch.Tensor: A 2D tensor of shape (num_nodes, N) containing the random walk matrix for ending up at the
                            same node at different depths N.
            - torch.Tensor: A 2D tensor of shape (num_nodes, num_nodes) containing the Laplacian matrix of the graph.
    """
    # Convert edge index into adjacency matrix
    edge_index = graph["edge_index"].numpy()
    num_nodes = np.max(edge_index) + 1
    adj_matrix = np.zeros((num_nodes, num_nodes))
    adj_matrix[edge_index[0], edge_index[1]] = 1
    adj_matrix[edge_index[1], edge_index[0]] = 1

    # Calculate degree matrix as row vector
    degrees = np.sum(adj_matrix, axis=0)
    randomwalk_matrix = []

    # First random walk (n=1) is just adjacency matrix / degrees vector
    randomwalk_matrix.append(adj_matrix / degrees)

    # Conduct random walk for N steps
    for _ in range(depth-1):
        randomwalk_i = np.zeros(adj_matrix.shape)

        # Divide probabilities by number of outgoing edges
        values = (randomwalk_matrix[-1].T / degrees).T

        # Redistribute position probabilities by the edges
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                col_vector = adj_matrix[:, j] * values[i, j]
                randomwalk_i[:, i] += col_vector

        randomwalk_matrix.append(randomwalk_i)

    # Calculate graph Laplacian: degree matrix - adjacency matrix
    laplacian = np.diag(np.sum(adj_matrix, axis=0)) - adj_matrix

    diagonal_elements = np.array([np.diag(rw_matrix) for rw_matrix in randomwalk_matrix])
    return torch.from_numpy(diagonal_elements), torch.from_numpy(laplacian)


if __name__ == '__main__':
    example_graph = torch.tensor(data=[[0, 0, 1, 1, 2, 3, 3], [1, 4, 4, 2, 3, 4, 5]])
    graph = {'edge_index': example_graph}
    lap_enc = lap_positional_encoding(graph, pos_encoding_dim=10)
    rw_enc = rw_positional_encoding(graph, depth=10)[0]
    print('Laplacian Eigenvector PE', lap_enc, '\n')
    print('RW-based PE', rw_enc, '\n')
