import math
import numpy as np
import matplotlib.pyplot as plt
import logging

logging.getLogger().setLevel(logging.INFO)

def f(x):
    return 3*x**2 - 4*x + 5

logging.info(f(3))

xs = np.arange(-5, 5, 0.25)
logging.info(xs)

ys = f(xs)
logging.info(ys)

plt.plot(xs, ys)
# plt.show()

h = 1e-9
x = 3
logging.info(f"derivative of f w.r.t x at x={x}= {(f(x+h)-f(x))/h}")


h = 1e-9
x = -3
logging.info(f"derivative of f w.r.t x at x={x}= {(f(x+h)-f(x))/h}")


h = 1e-9
x = 2/3
logging.info(f"derivative of f w.r.t x at x={x}= {(f(x+h)-f(x))/h}")