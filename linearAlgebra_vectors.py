import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 2 dimensional vector 
v2 = [3, -2]

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

# start at origin ( 0 ), then... for all 3 points
ax.plot([0, v3[0]], [0, v3[1]], [0, v3[2]], linewidth=3)

# make the plot look nicer 
ax.plot([0, 0], [0, 0], [-4, 4], 'k--')
ax.plot([0, 0], [-4, 4], [0, 0], 'k--')
ax.plot([-4, 4], [0, 0], [0, 0], 'k--')
plt.show()

# ewa_linearAlg -> element wise addiition using Linear Algebra 
# the below will not work within python, it would simply 
# concat the two vectors/list together making it a longer 
# vector/list

# [ 1, 2, 3 ] +  [ 3 ,4, 5 ] --> [ 1, 2, 3, 3 ,4, 5 ]

# ewa_python -> element wise addition using python

l1 = [ 1, 2, 3 ]
l2 = [ 3 ,4, 5 ]
[sum(x) for x in zip(l1,l2)]
# // [4,6,8] <-- results 

# ewa_python using numpy
# this way depends on the importing of a few modules
#   but it simplifys the implementation / syntax 

# two vectors in R2
v1 = np.array([3,-1])
v2 = np.array([2,4])
v3 = v1 + v2 

# plot the vector elements on graph 
# plotting the x variable but making sure the vector is in  standard position, by placing origin or tail at 0 before providing second coord/element to graph
#   finishing off by labeling the vector
plt.plot( [0, v1[0], [0,v1]], 'b', label='v1' )
plt.plot( [0, v2[0]] + v1[0], [0, v2[1]] + v1[1], 'r', label='v2')
plt.plot([0, v3[0]], [0, v3[1]], 'k', label='v1+v2')

plt.legend()
plt.axis('square')
plt.axis((-6, 6, -6, 6))
plt.grid()
plt.show()
