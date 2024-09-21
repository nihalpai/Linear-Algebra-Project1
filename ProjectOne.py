import numpy as np 
import json

def read_resistances(file_path):
    resistances = []
    with open(file_path, 'r') as file:
        data = json.load(file)
        for line in data:
            resistances.append((line["node1"], line["node2"], line["resistance"]))
    return resistances

def read_voltages(file_path):
    voltages = {}
    with open(file_path, 'r') as file:
        data = json.load(file)
        for line in data:
            voltages[line["node"]] = float(line["voltage"])
    return voltages

def compute_A_matrix(resistances, voltages):
    nodes = set()
    for node1, node2, _ in resistances:
        nodes.add(node1)
        nodes.add(node2)
    
    num_nodes = len(nodes)
    A = np.zeros((num_nodes, num_nodes))
    
    node_index = {node: idx for idx, node in enumerate(sorted(nodes))}
    
    for node1, node2, resistance in resistances:
        idx1 = node_index[node1]
        idx2 = node_index[node2]
        A[idx1][idx1] += 1 / resistance
        A[idx2][idx2] += 1 / resistance
        A[idx1][idx2] -= 1 / resistance
        A[idx2][idx1] -= 1 / resistance
    
    for node, voltage in voltages.items():
        idx = node_index[node]
        A[idx] = 0
        A[idx][idx] = 1
    
    return A, node_index

def lu_decomposition(A):
    n = A.shape[0]
    L = np.zeros_like(A)
    U = np.zeros_like(A)
    
    for i in range(n):
        L[i, i] = 1
        for j in range(i, n):
            U[i, j] = A[i, j] - L[i, :i] @ U[:i, j]
        for j in range(i+1, n):
            L[j, i] = (A[j, i] - L[j, :i] @ U[:i, i]) / U[i, i]
    
    return L, U

def forward_substitution(L, b):
    y = np.zeros_like(b)
    for i in range(len(b)):
        y[i] = b[i] - L[i, :i] @ y[:i]
    return y

def backward_substitution(U, y):
    x = np.zeros_like(y)
    for i in range(len(y) - 1, -1, -1):
        x[i] = (y[i] - U[i, i+1:] @ x[i+1:]) / U[i, i]
    return x

def compute_node_voltages(A, voltages):
    b = np.zeros(A.shape[0])
    for node, voltage in voltages.items():
        b[node_index[node]] = voltage
    
    L, U = lu_decomposition(A)
    y = forward_substitution(L, b)
    voltages_solution = backward_substitution(U, y)
    
    return voltages_solution

def compute_currents(resistances, voltages_solution, node_index):
    currents = {}
    for node1, node2, resistance in resistances:
        idx1 = node_index[node1]
        idx2 = node_index[node2]
        current = (voltages_solution[idx1] - voltages_solution[idx2]) / resistance
        currents[(node1, node2)] = current
    return currents

def write_output(file_path, voltages_solution, currents):
    with open(file_path, 'w') as file:
        file.write("Node Voltages:\n")
        for idx, voltage in enumerate(voltages_solution):
            file.write(f"Node {idx + 1}: {voltage:.2f} V\n")
        
        file.write("\nCurrents through each link:\n")
        for (node1, node2), current in currents.items():
            file.write(f"Link {node1} - {node2}: {current:.2f} A\n")

# Example usage
if __name__ == "__main__":
    resistance_file = '/Users/nihalpai/Documents/Linear Algebra/Linear-Algebra-Project1/node_resistances.json'  # Replace with your resistance file path
    voltage_file = '/Users/nihalpai/Documents/Linear Algebra/Linear-Algebra-Project1/node_voltages.json'  # Replace with your voltage file path
    output_file = '/Users/nihalpai/Documents/Linear Algebra/Linear-Algebra-Project1/answers.txt'  # Output file path
    
    resistances = read_resistances(resistance_file)
    voltages = read_voltages(voltage_file)
    A_matrix, node_index = compute_A_matrix(resistances, voltages)
    
    voltages_solution = compute_node_voltages(A_matrix, voltages)
    currents = compute_currents(resistances, voltages_solution, node_index)
    
    write_output(output_file, voltages_solution, currents)
    
    print("Output written to", output_file)
