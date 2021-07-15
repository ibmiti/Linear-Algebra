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


#  vector-scalar multiplcation ( aV )
#    multiplying a vector by a scalar ( a * v ) changes the length of the vector but does not change the direction of the vector 

# np will treat the array/list as a v ( vector )
v1 = np.array([3, -1])
l  = -.3
vlm = v1*l # scalar-modulated -> the scalar will effect the vectors length 

# plot them 
plt.plot([0, v1[0]], [0, v1[1]], 'b', label='v_1')
plt.plot([0, vlm[0]], [0, vlm[1]], 'r:', label='\label v_1')

plt.axis('square')
axlim = max([abs(max(v1)), abs(max(vlm))]) * 1.5 # dynamic axis lim
plt.axis(( -axlim, axlim, -axlim, axlim ))
plt.grid()
plt.show()

# vector-vector multiplication ( vv ) : the dot product
#   rule : both vectors/arrays/list must be of same length in order to carry out a successful dot-product ( vv ) operation. 
#   reason : [ 1, 2, 3, 4 ] * [ 2, 4, 5, null ] -> we cannot multiply an integer ( number ) by null / nothing

# here are the ways to compute the dot product betwixt v's 

# setting up the v's 
v1 = np.array([ 1, 2, 4, 5, 6, 7 ]) # length 6
v2 = np.array([ 0, -4, -3, 6, 5, 5 ]) # length 6 } must be same length - remember

# method 1 of computing dot-product ( vv )
dp1 = sum( np.multiply( v1, v2 ))

# method 2 ...
#  recommended way : it is the easiest way.
dp2 = np.dot( v1, 2 )

# method 3 // matrix multiplication
dp3 = np.matmul( v1, v2)

# method 4 // using loop
dp3 = 0 # initialize 

# loop over elements 
for i in range( 0 , len( v1 )):
    # multiply corresponding element and sum
    dp4 = dp3 + v1[i] * v2[i]

print(dp1, dp2, dp3, dp4)

# dot product is distributive 

#    quick examples of distribution 
# 5( 6 + 2 ) = 5 * 6 + 5 * 2 = 30 + 10 = 40

#  vs -> (5)(8) = 40 

## Distributive property 

# create random vectors 
# each vector will be equal length ( same-dimensionality )
#  they will have a length of 10 ( n = 10 )
n = 10
a = np.random.randn(n)
b = np.random.randn(n)
c = np.random.randn(n)

# the two results // { res1, res2 } -> should give same result
res1 = np.dot( a, ( b + c )) # res1 == res2
res2 = np.dot( a, b ) + np.dot( a, c ) # res2 == res1 

# compare them
print([ res1, res2 ])

## Associative property 

# create random vectors 
n = 5
a = np.random.randn(n)
b = np.random.randn(n)
c = np.random.randn(n)

# the two results 
res1 = np.dot( a, np.dot( b, c ))
res2 = np.dot( np.dot( a, b ) , c ) 

# compare them
# the product ( or result ) will widely vary and never be identical -> thus associative property on dot product of vectors is illegal
print(res1)
print(res2)

## Special cases where associate property works! 
# 1 ) one vector is the zeros vector ex : [ 0, 0, 0 ] 
# ( a * b ) + c = b ( a * c ) is true, if a is [0,0,0]
# 2 ) a==b==c -> if all vectors elements are same val.

## create 2 4x6 matrices of random numbers.
# Use a for-loop to compute dot products between corresponding columns 

A = np.random.randn( 4, 6 ) 
B = np.random.randn( 4, 6 )

print(A)
print(' ')
print(B)

dps = np.zeros( 6 ) 
for i in range( 6 ):
    dps[i] = np.dot( A[:,i], B[:,i] )

print(' ')
print(dps)













