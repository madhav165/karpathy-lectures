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

# create values and assign labels
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
logging.info(f"a = {a}")
logging.info(f"b = {b}")
logging.info(f"c = {c}")
e = a * b
e.label = 'e'
d = e + c
d.label = 'd'
f = Value(-2, label='f')
L = f * d
L.label = 'L'
logging.info(f"d = a * b + c = {d}")
logging.info(f'd._prev = {d._prev}')
logging.info(f'd._op = {d._op}')

# create graph object on final variable L
g = Graph(L)
logging.info(f'g = {g}')
logging.info(f'g.trace = {g.trace()}')

# assign gradients based on chain rule

# dL/dL = 1
L.grad = 1.0
# dL/df = d
f.grad = d.data
# dL/dd = f
d.grad = f.data
# dL/de = dL/dd * dd/de
e.grad = d.grad * 1.0
# dL/dc = dL/dd * dd/dc
c.grad = d.grad * 1.0
# dL/db = dL/de * de/db
b.grad = e.grad * a.data
# dL/da = dL/de * de/da
a.grad = e.grad * b.data

# display graph
gout = g.draw_dot()
gout.render('gout_value')