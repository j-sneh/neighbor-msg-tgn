from collections import deque

# given adj_list, determine partition of bipartite graph
def get_partition(adj_list):
    """
    Given an adjacency list of a bipartite graph, returns the partition of the graph using BFS.
    """
    adj_list = adj_list.copy()
    adj_list = [set(neighbor[0] for neighbor in neighbors) for neighbors in adj_list]

    n = len(adj_list)
    set1, set2 = set(), set()
    queue = deque()

    for start in range(n):
        if start not in set1 and start not in set2: # not visited
            queue.append((start, 0))  # (node, color)

            while queue:
                node, color = queue.popleft()

                if color == 0:
                    set1.add(node)
                else:
                    set2.add(node)

                for neighbor in adj_list[node]:
                    if (neighbor in set1 and color == 0) or (neighbor in set2 and color == 1):
                        # If we find a neighbor with the same color, it's not bipartite
                        raise ValueError("Graph is not bipartite")
                    if neighbor not in set1 and neighbor not in set2:
                        queue.append((neighbor, 1 - color))  # alternate color

    return set1, set2