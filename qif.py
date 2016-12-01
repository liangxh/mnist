#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

dt = 0.1
span = 1000. #ms

a = 0.02
b = 0.2
d = 8.

v_reset = -65.
v_thr = 35.

n_step = int(span / dt)
I = np.zeros(n_step)
I[int(200 / dt): int(700 / dt)] = 7. #pA

v = [-70., ]
u = [-14., ]
for i in range(1, n_step):
    v_prev = v[-1]
    u_prev = u[-1]
    if v_prev >= v_thr:
        v_t = v_reset
        u_t = u_prev + d
    else:
        dv = (0.04 * v_prev + 5.) * v_prev + 140. - u_prev + I[i]
        du = a * (b * v_prev - u_prev)

        v_t = v_prev + dv * dt
        u_t = u_prev + du * dt

    v.append(v_t)
    u.append(u_t)

plt.figure()
plt.plot(v, 'r')
plt.plot(u, 'b')
plt.show()
