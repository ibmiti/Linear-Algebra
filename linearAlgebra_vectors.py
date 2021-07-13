import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 2 dimensional vector 
v2 = [3,-2]

# 3 dimensional vector 
v3 = [ 4, -3, 2]

# to tranform a row-vector to column-vector ( or vice-versa ) -> transpose
v3t = np.transpose(v3)

# if the list ( arr ) has already been turned into -> numpy array we can do the following to transpose the vector from col-v <-> row-v
# v3.T
 
# plot them 

# this states : begin plot at origin ( 0 ), then plot the first el along the x-axis, do so with the y value, -> start at 0 then move along y by v2[1] which is equal to -2
plt.plot([0,v2[0]],[0,v2[1]])

plt.axis('equal')
plt.plot([-4, 4],[0, 0],'k--')