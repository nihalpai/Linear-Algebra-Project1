# import json


# def read_resistances_json(file_name):
#     resistances = {}
#     with open(file_name, 'r') as f:
#         data = json.load(f)
#         # for i in f:
#         #     return i
#         return data

#     #     for entry in data:
#     #         node1 = entry["node1"]
#     #         node2 = entry["node2"]
#     #         resistance = entry["resistance"]
#     #         resistances[(node1, node2)] = resistance
#     # return resistances

# resistances = read_resistances_json('Node_Resistance.json')
# print(resistances)

import numpy as np
import json
from scipy.linalg import lu, solve

def read_resistances(file_path):
    resistances = []
    with open(file_path, 'r') as file:
        data = json.load(file)
        for line in data:
            # node1, node2, resistance = line.split()
            resistances.append((line["node1"], line["node2"], line["resistance"]))
    return resistances

def read_voltages(file_path):
    voltages = {}
    with open(file_path, 'r') as file:
        data = json.load(file)
        for line in data:
            # node, voltage = line.split()
            voltages[line["node"]] = float(line["voltage"])
    return voltages

def compute_A_matrix(resistances, voltages):
    # Determine the number of nodes
    nodes = set()
    for node1, node2, _ in resistances:
        nodes.add(node1)
        nodes.add(node2)
    
    num_nodes = len(nodes)
    A = np.zeros((num_nodes, num_nodes))
    
    # Create a mapping from node number to index
    node_index = {node: idx for idx, node in enumerate(sorted(nodes))}
    
    # Fill the A matrix based on resistances
    for node1, node2, resistance in resistances:
        idx1 = node_index[node1]
        idx2 = node_index[node2]
        A[idx1][idx1] += 1 / resistance
        A[idx2][idx2] += 1 / resistance
        A[idx1][idx2] -= 1 / resistance
        A[idx2][idx1] -= 1 / resistance
    
    # Apply fixed voltages
    for node, voltage in voltages.items():
        idx = node_index[node]
        A[idx] = 0  # Set the row to zero
        A[idx][idx] = 1  # Set the diagonal to 1 for fixed voltage
        # You may also want to adjust the right-hand side vector if needed

    return A, node_index
def compute_node_voltages(A, voltages):
    # Create the right-hand side vector
    b = np.zeros(A.shape[0])
    for node, voltage in voltages.items():
        b[node_index[node]] = voltage
    
    # Solve for voltages using LU factorization
    P, L, U = lu(A)
    y = solve(L, np.dot(P.T, b))  # Solve Ly = Pb
    voltages_solution = solve(U, y)  # Solve Ux = y
    
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
    resistance_file = '/Users/aoke/Downloads/node_resistances.json'  # Replace with your resistance file path
    voltage_file = '/Users/aoke/Downloads/node_voltages.json'  # Replace with your voltage file path
    output_file = '/Users/aoke/Desktop/Linear/Proj_One_output.txt'  # Output file path
    
    resistances = read_resistances(resistance_file)
    voltages = read_voltages(voltage_file)
    A_matrix, node_index = compute_A_matrix(resistances, voltages)
    
    voltages_solution = compute_node_voltages(A_matrix, voltages)
    currents = compute_currents(resistances, voltages_solution, node_index)
    
    write_output(output_file, voltages_solution, currents)
    
    print("Output written to", output_file)