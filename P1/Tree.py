import numpy as np

# Define the nodes with their fixed voltages
nodes =  [
    {"node": 1, "voltage": 2.0},  # Node 1 fixed at 2V
    {"node": 2, "voltage": None},  # Nodes without fixed voltage are set to None
    {"node": 3, "voltage": None},
    {"node": 4, "voltage": None},  
    {"node": 5, "voltage": None},  
    {"node": 6, "voltage": None},   
    {"node": 7, "voltage": None},   
    {"node": 8, "voltage": 0.0},   # Grounded node
    {"node": 9, "voltage": 0.0},   # Grounded node
    {"node": 10, "voltage": 0.0},   # Grounded node
    {"node": 11, "voltage": 0.0},   # Grounded node
    {"node": 12, "voltage": 0.0},   # Grounded node
    {"node": 13, "voltage": 0.0},   # Grounded node
    {"node": 14, "voltage": 0.0},   # Grounded node
    {"node": 15, "voltage": 0.0}   # Grounded node
]

# Adjacency list for the tree
tree = {
    1: [2, 3],
    2: [1, 4, 5],
    3: [1, 6, 7],
    4: [2, 8, 9],
    5: [2, 10, 11],
    6: [3, 12, 13],
    7: [3, 14, 15],
    8: [4],
    9: [4],
    10: [5],
    11: [5],
    12: [6],
    13: [6],
    14: [7],
    15: [7]
}

def get_neighbors(node):
    """Return the neighbors of a given node in the tree."""
    return tree.get(node, [])

def get_fixed_voltages(nodes):
    """Return a dictionary of fixed voltages from the nodes list."""
    return {node['node']: node['voltage'] for node in nodes if node['voltage'] is not None}

def compute_A_matrix(resistances, fixed_voltages, n=15):
    A = np.zeros((n, n))  # Initialize A matrix
    b = np.zeros(n)       # Initialize b vector

    for node in range(1, n + 1):
        print(f'Processing node {node}')
        if node in fixed_voltages:
            A[node-1, node-1] = 1  # Set diagonal element for fixed voltage node
            b[node-1] = fixed_voltages[node]  # Set corresponding b value
        else:
            neighbors = get_neighbors(node)

            sum_conductance = 0
            for neighbor in neighbors:
                if (node, neighbor) in resistances:
                    resistance = resistances[(node, neighbor)]
                else:
                    resistance = resistances[(neighbor, node)]
                print(f'The resistance between nodes {node} and {neighbor} is {resistance}')
                conductance = 1 / resistance
                A[node-1, neighbor-1] = -conductance  # Set off-diagonal element
                sum_conductance += conductance
            A[node-1, node-1] = sum_conductance  # Set diagonal element
            b[node-1] = 0  # Set corresponding b value
    return A, b

# Example resistances
resistances = {
    (1, 2): 10,
    (1, 3): 15,
    (2, 4): 5,
    (2, 5): 10,
    (3, 6): 20,
    (3, 7): 25,
    (4, 8): 30,
    (4, 9): 35,
    (5, 10): 40,
    (5, 11): 45,
    (6, 12): 50,
    (6, 13): 55,
    (7, 14): 60,
    (7, 15): 65
}

# Get fixed voltages from the nodes list
fixed_voltages = get_fixed_voltages(nodes)

# Compute the A matrix and b vector
A, b = compute_A_matrix(resistances, fixed_voltages)
print('A =\n', A)
print('b =', b)

def lu_decomposition(A):
    """Perform LU decomposition of matrix A."""
    n = A.shape[0]
    L = np.zeros((n, n))  # Initialize L matrix
    U = np.zeros((n, n))  # Initialize U matrix

    for i in range(n):
        # Upper Triangular
        for j in range(i, n):
            U[i, j] = A[i, j] - np.dot(L[i, :i], U[:i, j])

        # Lower Triangular
        for j in range(i, n):
            if i == j:
                L[i, i] = 1  # Diagonal as 1
            else:
                L[j, i] = (A[j, i] - np.dot(L[j, :i], U[:i, i])) / U[i, i]

    return L, U

# Perform LU decomposition on the A matrix
L, U = lu_decomposition(A)
print("L =\n", L)
print("U =\n", U)

def forward_substitution(L, b):
    """Solve the equation L * y = b using forward substitution."""
    n = L.shape[0]
    y = np.zeros(n)  # Initialize y vector

    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])

    return y

def backward_substitution(U, y):
    """Solve the equation U * V = y using backward substitution."""
    n = U.shape[0]
    V = np.zeros(n)  # Initialize V vector

    for i in range(n - 1, -1, -1):
        V[i] = (y[i] - np.dot(U[i, i + 1:], V[i + 1:])) / U[i, i]

    return V

def compute_currents(V, resistances):
    currents = {}
    for (node1, node2), resistance in resistances.items():
        current = (V[node1 - 1] - V[node2 - 1]) / resistance
        currents[(node1, node2)] = current
    return currents

# Solve for node voltages
y = forward_substitution(L, b)
V = backward_substitution(U, y)

# Compute currents through each link
currents = compute_currents(V, resistances)

# Output the results
print("\nNode Voltages:")
for idx, voltage in enumerate(V, start=1):
    print(f"Node {idx}: {voltage:.4f} V")

print("\nCurrents through each link:")
for (node1, node2), current in currents.items():
    print(f"Current from Node {node1} to Node {node2}: {current:.6f} A")
