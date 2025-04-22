from collections import deque
class Node(object):
    """
    Node in a directed graph
    """
    def __init__(self, name=""):
        """
        Construct a new node, and initialize the list of parents and children.
        Each parent/child is represented by a (key, value) pair in dictionary, 
        where key is the parent/child's name, and value is an Node object.
        Args:
            name: a unique string identifier.
        """
        self.name = name
        self.parents = dict()
        self.children = dict()

    def add_parent(self, parent):
        """
        Args:
            parent: an Node object.
        """
        if not isinstance(parent, Node):
            raise ValueError("Parent must be an instance of Node class.")
        pname = parent.name
        self.parents[pname] = parent

    def add_child(self, child):
        """
        Args:
            child: an Node object.
        """
        if not isinstance(child, Node):
            raise ValueError("Parent must be an instance of Node class.")
        cname = child.name
        self.children[cname] = child


class BN(object):
    """
    Bayesian Network
    """
    def __init__(self):
        """
        Initialize the list of nodes in the graph.
        Each node is represented by a (key, value) pair in dictionary, 
        where key is the node's name, and value is an Node object
        """
        self.nodes = dict()

    def add_edge(self, edge):
        """
        Add a directed edge to the graph.
        
        Args:
            edge: a tuple (A, B) representing a directed edge A-->B,
                where A, B are two strings representing the nodes' names
        """
        (pname, cname) = edge

        ## construct a new node if it doesn't exist
        if pname not in self.nodes:
            self.nodes[pname] = Node(name=pname)
        if cname not in self.nodes:
            self.nodes[cname] = Node(name=cname)

        ## add edge
        parent = self.nodes.get(pname)
        child = self.nodes.get(cname) 
        parent.add_child(child)
        child.add_parent(parent)

    def print_graph(self):
        """
        Visualize the current graph.
        """
        print("-"*50)
        print("Bayes Network:")
        for nname, node in self.nodes.items():
            print("\tNode " + nname)
            print("\t\tParents: " + str(node.parents.keys()))
            print("\t\tChildren: " + str(node.children.keys()))
        print("-"*50)

    def is_dsep(self, start, end, observed):
        """
        Check whether start and end are d-separated given observed, using the Bayes Ball algorithm.
        Args:
            start: a string, name of the first query node
            end: a string, name of the second query node
            observed: a list of strings, names of the observed nodes.
        Returns:
            True if start and end are d-separated given observed, False otherwise.
        """
        if start not in self.nodes or end not in self.nodes:
            raise ValueError("Start or end node not found in the graph.")

        # Convert observed list to a set for faster lookup
        observed_set = set(observed)

        # Queue for BFS: (current_node, direction, is_blocked)
        # direction: 'up' (from child to parent) or 'down' (from parent to child)
        queue = deque()
        queue.append((self.nodes[start], 'down', False))

        # Visited set to avoid cycles: (node, direction)
        visited = set()

        while queue:
            current_node, direction, is_blocked = queue.popleft()

            # If we reach the end node and the path is not blocked, return False
            if current_node.name == end and not is_blocked:
                return False

            # Skip if this (node, direction) has already been visited
            if (current_node.name, direction) in visited:
                continue
            visited.add((current_node.name, direction))

            # Handle observed nodes
            if current_node.name in observed_set:
                # If the node is observed, the ball is blocked when coming from a parent
                if direction == 'down':
                    continue  # Blocked
                elif direction == 'up':
                    # Can go to children
                    for child in current_node.children.values():
                        queue.append((child, 'down', False))
            else:
                # If the node is not observed
                if direction == 'down':
                    # Can go to children
                    for child in current_node.children.values():
                        queue.append((child, 'down', is_blocked))
                    # Can go to parents (reverse direction)
                    for parent in current_node.parents.values():
                        queue.append((parent, 'up', is_blocked))
                elif direction == 'up':
                    # Can go to parents
                    for parent in current_node.parents.values():
                        queue.append((parent, 'up', is_blocked))
                    # Can go to children if not a collider
                    if not is_blocked:
                        for child in current_node.children.values():
                            queue.append((child, 'down', False))

        # If no path reaches the end node, return True (d-separated)
        return True


if __name__ == "__main__":
    # Test the BN class
    bn = BN()
    bn.add_edge(('A', 'B'))
    bn.add_edge(('B', 'C'))
    bn.print_graph()

    print(bn.is_dsep('A', 'C', ['B']))  # True (A and C are d-separated given B)
    print(bn.is_dsep('C', 'A', []))     # False (C and A are not d-separated)