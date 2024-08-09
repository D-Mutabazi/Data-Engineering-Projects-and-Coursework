import heapq

def read_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content

def parse_input(input_str):
    lines = input_str.strip().split("\n")
    nodes = {}
    
    for line in lines:
        print(f"Processing line: {line}")  # Debugging statement
        parts = line.strip("{}").split(";")
        
        if len(parts) < 1:
            print(f"Error: Line format is incorrect: {line}")
            continue
        
        coords_str = parts[0].strip("()")
        try:
            coords = tuple(map(int, coords_str.split(",")))
        except ValueError as e:
            print(f"Error parsing coordinates: {coords_str}")
            raise e
        
        neighbors = []
        if len(parts) > 1:  # Only if there are neighbors listed
            for neighbor in parts[1:]:
                try:
                    n_coords_cost = neighbor.strip("()").split(",")
                    if len(n_coords_cost) == 3:
                        n_coords = tuple(map(int, n_coords_cost[:2]))
                        cost = float(n_coords_cost[2])
                        neighbors.append((n_coords, cost))
                    else:
                        raise ValueError(f"Neighbor format is incorrect: {neighbor}")
                except ValueError as e:
                    print(f"Error parsing neighbor: {neighbor}")
                    raise e
        
        nodes[coords] = neighbors
    
    return nodes

def dijkstra_with_constraint(nodes, start, end, max_days):
    queue = [(0, start, 0)]  # (cost, current_node, days)
    visited = set()
    
    while queue:
        cost, current_node, days = heapq.heappop(queue)
        
        if current_node == end:
            return cost
        
        if current_node in visited:
            continue
        
        visited.add(current_node)
        
        for neighbor, travel_cost in nodes.get(current_node, []):
            if days + 1 <= max_days:
                heapq.heappush(queue, (cost + travel_cost, neighbor, days + 1))
    
    return float('inf')

# Path to your input file
file_path = '1.txt'

# Read the content from the file
input_data = read_file(file_path)

# Parse the input
nodes = parse_input(input_data)

# Define start and end coordinates
start_node = (0, 11)
end_node = (10, 0)
max_days = 10

# Find the minimum cost path
min_cost = dijkstra_with_constraint(nodes, start_node, end_node, max_days)
print("Minimum cost:", min_cost)
