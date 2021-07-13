import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 2 dimensional vector 
v2 = [3,-2]

# 3 dimensional vector 
v3 = [ 4, -3, 2]

# to tranform a row-vector to column-vector ( or vice-versa ) -> transpose
v3t = np.transpose(v3)

# plot them 
plt.plot([0,v2[0]],[0,v2[1]])
plt.axis('equal')
plt.plot([-4, 4],[0, 0],'k--')