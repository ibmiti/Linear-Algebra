import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 2 dimensional vector 
v2 = [3,-2]

# 3 dimensional vector 
# will use this 3d-v in creating a 3dimensional vector 
# v3 has a dimensionality of 3 -> size of 3  || length of 3
# v3[0] -> v3[x] , v3[1] -> v3[y],  v3[2] -> v3[3] ( 3rd dimension )
v3 = [ 4, -3, 2] 

# to tranform a row-vector to column-vector ( or vice-versa ) -> transpose
v3t = np.transpose(v3)

# if the list ( arr ) has already been turned into -> numpy array we can do the following to transpose the vector from col-v <-> row-v
# v3.T
 
# plot them 

# this is a 2d-Vector 
# this states : begin plot at origin ( 0 ), then plot the first el along the x-axis, do so with the y value, -> start at 0 then move along y by v2[1] which is equal to -2
plt.plot([0,v2[0]],[0,v2[1]])
plt.axis('equal')
plt.plot([-4, 4],[0, 0],'k--')
plt.plot([0,0],[-4, 4], 'k--')
plt.grid()
plt.axis((-4, 4, -4, 4))
plt.show()

# plotting a 3d Vector

fig = plt.figure(figsize=plt.figaspect(1))
ax  = fig.gca(projection='3d')

# start at origin ( 0 ), then...
ax.plot([0, v3[0]], [0, v3[1]])