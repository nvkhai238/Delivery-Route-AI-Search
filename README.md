# Delivery-Route-AI-Search
1.  Student Info

- Name: Ninh Van Khai
- ID: 23127060

---- Problem: Delivery Route Optimization with Time Windows ----

2. Problem Overview

- Plan an optimal delivery route for (1 truck) visiting (5 customers).
- Each customer has a (time window).
- Goal: Minimize total distance while respecting time windows and returning to depot.

3. ---- How to run program ----

- Step 1: pip install networkx matplotlib numpy
  -> This line helps us install library needed for program
  -> If you encounter rendering/layout issues (optional): pip install pygraphviz 
- Step 2: Run program with: python main.py

4. Algorithms Used

- A\* Search:

  - Uses `f(n) = g(n) + h(n)`
  - Guarantees optimal solution with admissible heuristic

- Greedy Best-First Search:
  - Uses only `h(n)` (nearest unvisited)
  - Faster but **not guaranteed optimal**

Note: Trap Design

- Customers with tight or late time windows placed near the depot.
- Greedy may choose early but invalid options.
- A\* avoids this by considering time feasibility and full cost.

5. Performance Metrics

- Compared:

  - Total Distance
  - Nodes Expanded
  - Max Frontier Size
  - Execution Time

- A\*: Optimal but slower
- Greedy: Faster but risk of failure/suboptimality

6. Outputs

- `initial_graph.png`: Delivery network with time windows
- `astar_solution.png`, `greedy_solution.png`: Route visualizations
- `*_search_tree.png`: Search tree for each algorithm
