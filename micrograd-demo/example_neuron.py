import math
import numpy as np
import matplotlib.pyplot as plt
import logging
from graph import Graph
from value import Value

logging.getLogger().setLevel(logging.INFO)

# change to directory of the file
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# inputs x1, x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')

# weights w1, w2
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')

# bias
b = Value(6.8813735870195432, label='b')

# neuron model
x1w1 = x1 * w1; x1w1.label = 'x1*w1'
x2w2 = x2 * w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label='n'
o = n.tanh(); o.label='o'

# calculating the grad one node at a time 
# in reverse orderusing topological graph
o.backward()

print(o)

g = Graph(o)
gout = g.draw_dot()
gout.render('gout_neuron')