import math
import numpy as np
import matplotlib.pyplot as plt
import logging

logging.getLogger().setLevel(logging.INFO)

a = 2
b = -3
c = 10

d = a * b + c

h = 1e-9

d1 = a * b + c

d2 = (a + h) * b + c
logging.info(f'derivative of d w.r.t a at a={a} = {(d2-d1)/h}')

d2 = a * (b + h) + c
logging.info(f'derivative of d w.r.t b at b={b} = {(d2-d1)/h}')

d2 = a * b + (c + h)
logging.info(f'derivative of d w.r.t c at c={c} = {(d2-d1)/h}')