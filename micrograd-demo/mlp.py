from layer import Layer

class MLP:

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [l for layer in self.layers for l in layer.parameters()]
        
# x = [2.0, 3.0]
# m = MLP(3, [4, 4, 1])
# print(m(x))