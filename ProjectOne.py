import numpy as np
import json

"""Load data from the json file given and map keys."""
def load_json_data(file_path, key_mapping):
    with open(file_path, 'r') as f:
        data = json.load(f)
        return [
                    {
                        new_key: entry[old_key]
                        for old_key, new_key in key_mapping.items()
                    }
                    for entry in data
                ]
 

"""Assemble matrix A and vector b for Ax = b."""
def assemble_system_matrices(resistance_list, voltage_nodes):
    
     # Identify all unique nodes 
    nodes = set()
    for item in resistance_list:
        nodes.update([item['node_start'], item['node_end']])  # Collect all nodes connected by resistors

    nodes = sorted(nodes)
    node_indices = {node: idx for idx, node in enumerate(nodes)}  # Map each node to an index
    n = len(nodes)

    # Initialize A matrix and b vector
    A_matrix = np.zeros((n, n))  # Start with zero matrices
    b_vector = np.zeros(n)

    # Build the A matrix using conductances
    for item in resistance_list:
        i = node_indices[item['node_start']]
        j = node_indices[item['node_end']]
        conductance = 1 / item['value']
        A_matrix[i, i] += conductance    # Add conductance to diagonal elements
        A_matrix[j, j] += conductance
        A_matrix[i, j] -= conductance    # Subtract conductance from off-diagonal elements
        A_matrix[j, i] -= conductance

    # Apply voltage constraints and build b vector
    for node_info in voltage_nodes:
        idx = node_indices[node_info['node']]
        voltage = node_info['value']
        A_matrix[idx, :] = 0            # Zero out the entire row for this node
        A_matrix[idx, idx] = 1          # Set the diagonal element to 1
        b_vector[idx] = voltage         # Set the known voltage in b

    return A_matrix, b_vector, node_indices


"""Perform LU factorization on matrix A."""
def perform_lu_factorization(A):

    n = A.shape[0]
    L = np.eye(n)          # Initialize L as identity matrix
    U = A.copy()           # Copy A into U

    for i in range(n):
        pivot = U[i, i]
        if pivot == 0:
            raise ZeroDivisionError(f"Zero pivot encountered at index {i}")  # Can't divide by zero
        
        for j in range(i+1, n):
            factor = U[j, i] / pivot       # Compute the factor to eliminate U[j, i]
            L[j, i] = factor               # Store factor in L
            U[j, i:] -= factor * U[i, i:]  # Update the rest of the row
            U[j, i] = 0                    # Set the lower part to zero
  
    return L, U


"""Solve the system Ax = b using forward and backward substitution."""
def solve_linear_system(L, U, b):

    # Forward substitution to solve Ly = b
    y = np.zeros_like(b)
    for i in range(len(b)):
        y[i] = b[i] - L[i, :i] @ y[:i]    # Calculate y using previous y values

    # Backward substitution to solve Ux = y
    x = np.zeros_like(y)
    for i in range(len(b)-1, -1, -1):
        x[i] = (y[i] - U[i, i+1:] @ x[i+1:]) / U[i, i]  # Calculate x using known y and x values
    return x

"""Calculate currents through each link using Ohm's Law."""
def calculate_currents(resistance_list, voltages, node_indices):
    currents = {}

    for item in resistance_list:
        idx_i = node_indices[item['node_start']]
        idx_j = node_indices[item['node_end']]
        voltage_diff = voltages[idx_i] - voltages[idx_j]    # Find voltage difference between nodes
        current = voltage_diff / item['value']              # Compute current I = V/R
        currents[(item['node_start'], item['node_end'])] = current

    return currents


"""Write the node voltages and currents to an output file."""
def write_results(output_path, voltages, currents, node_indices):    
    idx_to_node = {idx: node for node, idx in node_indices.items()}
    
    with open(output_path, 'w') as f:
        f.write("Node Voltages:\n")

        for idx, voltage in enumerate(voltages):
            f.write(f"Node {idx_to_node[idx]}: {voltage:.4f} V\n")  # Write voltage of each node
        f.write("\nCurrents through each link:\n")
        
        for (node_start, node_end), current in currents.items():
            f.write(f"Link {node_start} - {node_end}: {current:.6f} A\n")  # Write current through each link



if __name__ == "__main__":
    # Replace with your actual file paths
    resistance_file = '/Users/nihalpai/Documents/Linear Algebra/Linear-Algebra-Project1/node_resistances.json'
    voltage_file = '/Users/nihalpai/Documents/Linear Algebra/Linear-Algebra-Project1/node_voltages.json'
    output_file = '/Users/nihalpai/Documents/Linear Algebra/Linear-Algebra-Project1/output_results.txt'

    # Load resistances and voltage constraints
    resistance_list = load_json_data(resistance_file, {'node1': 'node_start', 'node2': 'node_end', 'resistance': 'value'})
    voltage_nodes = load_json_data(voltage_file, {'node': 'node', 'voltage': 'value'})

    # Assemble system matrices
    A_matrix, b_vector, node_indices = assemble_system_matrices(resistance_list, voltage_nodes)

    # Perform LU factorization and solve the system
    L_matrix, U_matrix = perform_lu_factorization(A_matrix)
    voltage_solution = solve_linear_system(L_matrix, U_matrix, b_vector)

    # Calculate currents through each link
    current_dict = calculate_currents(resistance_list, voltage_solution, node_indices)

    # Write the results to a file
    write_results(output_file, voltage_solution, current_dict, node_indices)

    print(f"Computation completed. Results have been written to {output_file}")
