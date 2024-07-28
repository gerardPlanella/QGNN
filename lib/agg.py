import torch

#Maybe aggregation should be permutation invariant
def aggregate_messages_by_product(h_i, sender, receiver, m_ij):
    # Initialize aggregated_messages to identity matrices
    identity = torch.eye(h_i.size(1), dtype=m_ij.dtype, device=h_i.device)
    aggregated_messages = identity.repeat(h_i.size(0), 1, 1)

    # Iterate over each message
    for i in range(m_ij.size(0)):
        send_idx = sender[i]
        recv_idx = receiver[i]
        
        # Skip if sender and receiver are the same (self-message)
        if send_idx == recv_idx:
            continue
        
        # Multiply the message to the current aggregated message for the receiver
        # Update in the new tensor to avoid in-place operations
        aggregated_messages[recv_idx] =  aggregated_messages[recv_idx] @ m_ij[i]

    return aggregated_messages

def aggregate_messages_by_product_optimized(h_i, sender, receiver, m_ij):
    N, D, _ = h_i.shape

    # Initialize aggregated_messages with identity matrices
    aggregated_messages = torch.eye(D, dtype=m_ij.dtype, device=h_i.device).repeat(N, 1, 1)

    # Loop over each unique receiver
    for recv_idx in torch.unique(receiver):
        # Find indices where this receiver is the target
        target_indices = torch.where(receiver == recv_idx)[0]

        # Skip if no messages are sent to this receiver
        if len(target_indices) == 0:
            continue

        # Initialize a temporary product matrix for this receiver
        temp_product = torch.eye(D, dtype=m_ij.dtype, device=h_i.device)

        # Iterate over each message sent to this receiver
        for idx in target_indices:
            temp_product = temp_product @ m_ij[idx]

        # Update the aggregated messages for this receiver
        aggregated_messages[recv_idx] = temp_product

    return aggregated_messages




# Test setup
D = 2  # Dimension of the messages
N = 3  # Number of nodes

# Define messages
m_ij = torch.tensor([
    [[0.1, 0], [0, 0.1j]],  # Identity
    [[0.2 + 0.23j, 0], [0, 0.2]],  # Scaling by 2
    [[0, 0.1j], [0.1 + 3j, 1 + 1j]],  # Swap
    [[0.1, 0.1], [0.1, 0.1j]]   # Ones
], dtype=torch.complex128)

# Define sender and receiver for each message
sender = torch.tensor([0, 0, 1, 2])
receiver = torch.tensor([1, 2, 2, 1])

# Node features (not used in this example but required for the function)
h_i = torch.randn(N, D, D)

# Run the aggregation function
aggregated_messages = aggregate_messages_by_product_optimized(h_i, sender, receiver, m_ij)

# Expected results
expected_node_0 = torch.eye(D, dtype=torch.complex128)
expected_node_1 = m_ij[0] @ m_ij[3]
expected_node_2 = m_ij[1] @ m_ij[2]

# Assert statements to verify the results
assert torch.allclose(aggregated_messages[0], expected_node_0), "Aggregation for Node 0 is incorrect"
assert torch.allclose(aggregated_messages[1], expected_node_1), "Aggregation for Node 1 is incorrect"
assert torch.allclose(aggregated_messages[2], expected_node_2), "Aggregation for Node 2 is incorrect"

print("All tests passed.")
