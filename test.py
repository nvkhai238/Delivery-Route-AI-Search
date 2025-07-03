#====================
# MSSV: 23127060
# Name: Ninh Van Khai
#====================
import networkx as nx
import numpy as np
import heapq
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta
import matplotlib as mpl

# ============= CONFIGURATION =============
N_CUSTOMERS = 5
DEPOT_ID = 0
SPEED = 30  # km/h
START_TIME = datetime(2023, 1, 1, 8, 0)  # Start at 8:00 AM
DISTANCE_METRIC = 'euclidean'  # 'euclidean' or 'manhattan'

# ============= FIXED COORDINATES =============
fixed_pos = {
    DEPOT_ID: (0, 0),  # Depot at origin
    1: (2, 2),        # Close to depot (trap for Greedy)
    2: (10, 0),       # Far along x-axis, early window
    3: (3, 3),        # Close to customer 1, very early window
    4: (10, 10),      # Far away, mid window
    5: (8, 10)        # Far in y-direction, mid window
}

# Time windows (start, end) as datetime objects
time_windows = {
    1: (datetime(2025, 1, 1, 11, 0), datetime(2025, 1, 1, 11, 30)),  # Late window (trap)
    2: (datetime(2025, 1, 1, 8, 20), datetime(2025, 1, 1, 8, 40)),   # Very early, tight
    3: (datetime(2025, 1, 1, 8, 5), datetime(2025, 1, 1, 8, 15)),    # Very early, tight
    4: (datetime(2025, 1, 1, 9, 30), datetime(2025, 1, 1, 10, 0)),   # Mid window
    5: (datetime(2025, 1, 1, 9, 0), datetime(2025, 1, 1, 9, 30))     # Mid window
}

# ============= CREATE GRAPH =============
G = nx.complete_graph(len(fixed_pos))
for u, v in G.edges():
    dx = fixed_pos[u][0] - fixed_pos[v][0]
    dy = fixed_pos[u][1] - fixed_pos[v][1]
    
    if DISTANCE_METRIC == 'euclidean':
        distance = np.sqrt(dx**2 + dy**2)
    else:  # manhattan
        distance = abs(dx) + abs(dy)
        
    G[u][v]['distance'] = distance
    G[u][v]['time'] = distance / SPEED * 60  # in minutes

# ============= VISUALIZE INITIAL GRAPH =============
def plot_initial_graph(G, pos, title, filename):
    """Plot and save the initial graph with depot, customers, and time windows"""
    plt.figure(figsize=(10, 8))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color='lightblue')
    nx.draw_networkx_nodes(G, pos, nodelist=[DEPOT_ID], node_size=1000, node_color='green')
    
    # Create labels with time windows
    labels = {}
    for node in G.nodes():
        if node == DEPOT_ID:
            labels[node] = f"Depot\nStart: {START_TIME.strftime('%H:%M')}"
        else:
            start = time_windows[node][0].strftime('%H:%M')
            end = time_windows[node][1].strftime('%H:%M')
            labels[node] = f"C{node}\n{start}-{end}"
    
    # Draw labels and edges
    nx.draw_networkx_labels(G, pos, labels, font_size=9)
    nx.draw_networkx_edges(G, pos, edge_color='lightgray', alpha=0.5)
    
    # Add legend
    plt.scatter([], [], c='green', s=200, label='Depot')
    plt.scatter([], [], c='lightblue', s=150, label='Customers')
    plt.legend(loc='best')
    
    plt.title(title, fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {filename}")

# ============= ADMISSIBLE HEURISTIC =============
def heuristic(node, unvisited, pos):
    """Admissible heuristic using Euclidean/Manhattan distance"""
    if not unvisited:
        # Return to depot
        dx = pos[node][0] - pos[DEPOT_ID][0]
        dy = pos[node][1] - pos[DEPOT_ID][1]
        return abs(dx) + abs(dy) if DISTANCE_METRIC == 'manhattan' else np.sqrt(dx**2 + dy**2)
    
    min_dist = float('inf')
    for c in unvisited:
        dx = pos[node][0] - pos[c][0]
        dy = pos[node][1] - pos[c][1]
        
        if DISTANCE_METRIC == 'manhattan':
            dist = abs(dx) + abs(dy)
        else:  # euclidean
            dist = np.sqrt(dx**2 + dy**2)
            
        if dist < min_dist:
            min_dist = dist
            
    return min_dist

# ============= TIME WINDOW HANDLING =============
def handle_time_window(arrival_time, customer):
    """Handle waiting time and check time window feasibility"""
    if customer == DEPOT_ID:
        return arrival_time, True
    
    start, end = time_windows[customer]
    
    # Handle early arrival (wait until window opens)
    if arrival_time < start:
        arrival_time = start
        
    # Check if within time window
    valid = start <= arrival_time <= end
    return arrival_time, valid

# ============= GENERIC SEARCH ALGORITHM =============
def tsp_search(G, pos, algorithm='astar'):
    start = DEPOT_ID
    all_customers = set(range(1, N_CUSTOMERS + 1))
    initial_state = (start, frozenset(), START_TIME)
    g_score = {initial_state: 0}
    frontier = []
    heapq.heappush(frontier, (0, 0, initial_state))
    came_from = {}
    nodes_expanded = 0
    max_frontier = 1
    count = 1
    search_tree = nx.DiGraph()
    search_tree.add_node(initial_state, label=f"0:D", depth=0)
    
    start_time = time.time()
    
    while frontier:
        _, _, state = heapq.heappop(frontier)
        current, visited, current_time = state
        nodes_expanded += 1

        # Goal check: back at depot with all customers visited
        if current == start and visited == all_customers:
            # Reconstruct path
            path = []
            times = []
            s = state
            while s in came_from:
                path.append(s[0])
                times.append(s[2])
                s = came_from[s]
            path.append(start)
            times.append(START_TIME)
            path.reverse()
            times.reverse()
            elapsed = time.time() - start_time
            return path, times, g_score[state], nodes_expanded, max_frontier, search_tree, elapsed

        for neighbor in G.neighbors(current):
            # Skip depot if not all customers visited
            if neighbor == start and visited != all_customers:
                continue
                
            # Skip already visited customers
            if neighbor != start and neighbor in visited:
                continue
                
            # Calculate travel time and arrival
            travel_time = timedelta(minutes=G[current][neighbor]['time'])
            arrival_time = current_time + travel_time
            
            # Handle time window constraints
            adjusted_time, valid = handle_time_window(arrival_time, neighbor)
            if not valid:
                continue
                
            # Update visited set
            new_visited = visited if neighbor == start else visited | {neighbor}
            new_state = (neighbor, new_visited, adjusted_time)
            
            # Calculate new distance
            tentative_g = g_score[state] + G[current][neighbor]['distance']
            
            if new_state not in g_score or tentative_g < g_score[new_state]:
                g_score[new_state] = tentative_g
                h_val = heuristic(neighbor, all_customers - new_visited, pos)
                f_val = tentative_g + h_val if algorithm == 'astar' else h_val
                heapq.heappush(frontier, (f_val, count, new_state))
                count += 1
                came_from[new_state] = state
                
                # Create descriptive label for search tree
                visited_str = ''.join(str(c) for c in sorted(new_visited))
                time_str = new_state[2].strftime('%H:%M')
                label = f"{count-1}:{neighbor}({visited_str})\n{time_str}"
                
                # Add node to search tree with depth information
                depth = search_tree.nodes[state]['depth'] + 1
                search_tree.add_node(new_state, label=label, depth=depth)
                search_tree.add_edge(state, new_state)
                
                max_frontier = max(max_frontier, len(frontier))

    elapsed = time.time() - start_time
    return None, None, float('inf'), nodes_expanded, max_frontier, search_tree, elapsed

# ============= VISUALIZATION FUNCTIONS =============
def plot_search_tree(tree, title, filename):
    plt.figure(figsize=(15, 10))
    
    # Create layout based on depth
    depth = nx.get_node_attributes(tree, 'depth')
    pos = {}
    for node, d in depth.items():
        # Get all nodes at this depth
        same_depth = [n for n, dep in depth.items() if dep == d]
        x_positions = np.linspace(0, 1, len(same_depth))
        idx = same_depth.index(node)
        pos[node] = (x_positions[idx], -d)
    
    labels = nx.get_node_attributes(tree, 'label')
    nx.draw(tree, pos, with_labels=True, labels=labels, node_size=2500, node_color='lightblue',
            font_size=8, font_weight='bold', edge_color='gray', arrowsize=15)
    plt.title(title, fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {filename}")

def plot_solution_path(G, pos, path, times, title, filename):
    plt.figure(figsize=(10, 8))
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color='lightblue')
    nx.draw_networkx_nodes(G, pos, nodelist=[DEPOT_ID], node_size=1000, node_color='green')
    
    # Create labels with times
    labels = {}
    for i, node in enumerate(path):
        time_str = times[i].strftime('%H:%M')
        labels[node] = f"{'D' if node == DEPOT_ID else node}\n{time_str}"
    
    # Draw all labels
    nx.draw_networkx_labels(G, pos, labels, font_size=9)
    
    # Draw all edges as light gray
    nx.draw_networkx_edges(G, pos, edge_color='lightgray', alpha=0.5)
    
    # Highlight solution path with red arrows
    path_edges = list(zip(path[:-1], path[1:]))
    
    # Create a directed graph for the solution path
    path_graph = nx.DiGraph()
    for u, v in path_edges:
        path_graph.add_edge(u, v)
    
    # Draw the entire path at once
    nx.draw_networkx_edges(
        path_graph, 
        pos,
        edge_color='red',
        width=3,
        arrows=True,
        arrowstyle='-|>',
        arrowsize=20,
        node_size=0  # Make sure arrows connect properly
    )
    
    plt.title(title, fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {filename}")

# ============= MAIN EXECUTION =============
if __name__ == "__main__":
    print("===== DELIVERY ROUTE PLANNING =====")
    print(f"Planning route for {N_CUSTOMERS} customers with time windows")
    print(f"Using {DISTANCE_METRIC} distance metric")
    results = {}

    # Visualize initial graph
    print("\n===== VISUALIZING INITIAL GRAPH =====")
    plot_initial_graph(G, fixed_pos, "Initial Delivery Network", "initial_graph.png")

    # Run A* search
    print("\n===== A* SEARCH =====")
    (astar_path, astar_times, astar_cost, astar_nodes, 
     astar_frontier, astar_tree, astar_time) = tsp_search(G, fixed_pos, 'astar')
     
    if astar_path:
        print(f"Path: {' -> '.join('D' if n == DEPOT_ID else str(n) for n in astar_path)}")
        print(f"Cost: {astar_cost:.2f} km | Nodes Expanded: {astar_nodes} | "
              f"Max Frontier: {astar_frontier} | Time: {astar_time:.4f}s")
        plot_solution_path(G, fixed_pos, astar_path, astar_times, 
                          f"A* Delivery Route ({DISTANCE_METRIC.capitalize()})", 
                          f"astar_{DISTANCE_METRIC}_solution.png")
        plot_search_tree(astar_tree, "A* Search Tree", f"astar_{DISTANCE_METRIC}_search_tree.png")
        results['astar'] = {
            'Cost': astar_cost,
            'Nodes Expanded': astar_nodes,
            'Max Frontier': astar_frontier,
            'Time (s)': astar_time
        }

    # Run Greedy search
    print("\n===== GREEDY SEARCH =====")
    (greedy_path, greedy_times, greedy_cost, greedy_nodes, 
     greedy_frontier, greedy_tree, greedy_time) = tsp_search(G, fixed_pos, 'greedy')
     
    if greedy_path:
        print(f"Path: {' -> '.join('D' if n == DEPOT_ID else str(n) for n in greedy_path)}")
        print(f"Cost: {greedy_cost:.2f} km | Nodes Expanded: {greedy_nodes} | "
              f"Max Frontier: {greedy_frontier} | Time: {greedy_time:.4f}s")
        plot_solution_path(G, fixed_pos, greedy_path, greedy_times, 
                          f"Greedy Delivery Route ({DISTANCE_METRIC.capitalize()})", 
                          f"greedy_{DISTANCE_METRIC}_solution.png")
        plot_search_tree(greedy_tree, "Greedy Search Tree", f"greedy_{DISTANCE_METRIC}_search_tree.png")
        results['greedy'] = {
            'Cost': greedy_cost,
            'Nodes Expanded': greedy_nodes,
            'Max Frontier': greedy_frontier,
            'Time (s)': greedy_time
        }

    # Print statistics
    print("\n===== PERFORMANCE COMPARISON =====")
    print(f"{'Metric':<15} | {'A*':<12} | {'Greedy':<12}")
    print("-" * 40)
    for metric in ['Cost', 'Nodes Expanded', 'Max Frontier', 'Time (s)']:
        a_val = results.get('astar', {}).get(metric, 'N/A')
        g_val = results.get('greedy', {}).get(metric, 'N/A')
        print(f"{metric:<15} | {a_val if isinstance(a_val, str) else a_val:<12.2f} | "
              f"{g_val if isinstance(g_val, str) else g_val:<12.2f}")
