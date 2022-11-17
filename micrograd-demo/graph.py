from graphviz import Digraph

class Graph:

    def __init__(self, root):
        self.root = root

    def __repr__(self):
        return f"Graph(root={self.root}))"

    def trace(self):
        nodes, edges = set(), set()
        
        def build(v):
            if v not in nodes:
                nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
        
        build(self.root)
        return nodes, edges

    def draw_dot(self):
        dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})

        nodes, edges = self.trace()

        for n in nodes:
            uid = str(id(n))
            dot.node(name = uid, label = f'{n.label} | data {n.data:>.4f} | grad {n.grad:>.4f}', shape = 'record')
            if n._op:
                dot.node(name = uid + n._op, label = n._op)
                dot.edge(uid + n._op, uid)

        for n1, n2 in edges:
            dot.edge(str(id(n1)), str(id(n2)) + n2._op)

        return dot
