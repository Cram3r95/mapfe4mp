import numpy as np
from csaps import csaps
import matplotlib.pyplot as plt
import pdb

np.random.seed(1234)
theta = np.linspace(0, 2*np.pi, 35)
x = np.cos(theta) + np.random.randn(35) * 0.1
y = np.sin(theta) + np.random.randn(35) * 0.1
data = [x, y]
theta_i = np.linspace(0, 2*np.pi, 200)
pdb.set_trace()
data_i = csaps(theta, data, theta_i, smooth=0.95)
xi = data_i[0, :]
yi = data_i[1, :]

fig, ax = plt.subplots(figsize=(8, 8))
plt.plot(x, y, ':o', xi, yi, '*')
plt.savefig("csaps.png", bbox_inches='tight', facecolor=fig.get_facecolor(), 
                    edgecolor='none', pad_inches=0)
