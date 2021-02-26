#!/usr/bin/env python
# coding: utf-8

# # Numpy Quickstart tutorial

# ## Pre-requisites
# 
# * Python
# * Installation - https://scipy.org/install.html

# ## The Basics
# 
# - Homogeneous multidimensional array.
# - Table of elements (usually numbers).
# - All of the same type.
# - Indexed by a tuple of non-negative integers.
# - Dimensions are called axes.

# Example:
# Coordinates of a point in 3D space [1, 2, 1]:
# 
# - Has one axis.
# - Axis has 3 elements
# - It has a length of 3.
# 
# In the example pictured below, the array has 2 axes. The first axis has a length of 2, the second axis has a length of 3.

# In[1]:


[[ 1., 0., 0.],
 [ 0., 1., 2.]]


# Important attributes of an ndarray object are:
# 
# **ndarray.ndim**
# the number of axes (dimensions) of the array.
# 
# **ndarray.shape**
# the dimensions of the array. This is a tuple of integers indicating the size of the array in each dimension. For a matrix with n rows and m columns, shape will be (n,m). The length of the shape tuple is therefore the number of axes, ndim.
# 
# **ndarray.size**
# the total number of elements of the array. This is equal to the product of the elements of shape.
# 
# **ndarray.dtype**
# an object describing the type of the elements in the array. One can create or specify dtype’s using standard Python types. Additionally NumPy provides types of its own. numpy.int32, numpy.int16, and numpy.float64 are some examples.
# 
# **ndarray.itemsize**
# the size in bytes of each element of the array. For example, an array of elements of type float64 has itemsize 8 (=64/8), while one of type complex32 has itemsize 4 (=32/8). It is equivalent to ndarray.dtype.itemsize.
# 
# **ndarray.data**
# the buffer containing the actual elements of the array. Normally, we won’t need to use this attribute because we will access the elements in an array using indexing facilities.

# ### An example

# In[2]:


import numpy as np
a = np.arange(15).reshape(3, 5)
a


# In[3]:


a.shape


# In[4]:


a.ndim


# In[5]:


a.dtype.name


# In[6]:


a.itemsize


# In[7]:


a.size


# In[8]:


type(a)


# In[9]:


b = np.array([6,7,8])


# In[10]:


type(b)


# ### Array Creation
# 
# - Create an array from a regular Python list or tuple using the array function.
# - The type of the resulting array is deduced from the type of the elements in the sequences.

# In[11]:


import numpy as np
a = np.array([2,3,4])
a


# In[12]:


a.dtype


# In[13]:


b = np.array([1.2, 3.5, 5.1])
b.dtype


# A frequent error consists in calling array with multiple numeric arguments, rather than providing a single list of numbers as an argument.

# In[14]:


# a = np.array(1,2,3,4)    # WRONG
a = np.array([1,2,3,4])  # RIGHT


# - array transforms sequences of sequences into two-dimensional arrays
# - sequences of sequences of sequences into three-dimensional arrays, and so on.

# In[15]:


b = np.array([(1.5,2,3), (4,5,6)])
b


# The type of the array can also be explicitly specified at creation time:

# In[16]:


c = np.array( [ [1,2], [3,4] ], dtype = complex )
c


# Often, the elements of an array are originally unknown, but its size is known. Hence, NumPy offers several functions to create arrays with initial placeholder content. These minimize the necessity of growing arrays, an expensive operation.
# 
# The function zeros creates an array full of zeros, the function ones creates an array full of ones, and the function empty creates an array whose initial content is random and depends on the state of the memory. By default, the dtype of the created array is float64.

# In[17]:


np.zeros( (3,4) )


# In[18]:


np.ones( (2,3,4), dtype=np.int16 ) # dtype can also be specified


# In[19]:


np.empty( (2,3) ) # uninitialized, output may vary


# To create sequences of numbers, NumPy provides a function analogous to range that returns arrays instead of lists.

# In[20]:


np.arange( 10, 30, 5 )


# In[21]:


np.arange( 0, 2, 0.3 ) # it accepts float arguments


# When arange is used with floating point arguments, it is generally not possible to predict the number of elements obtained, due to the finite floating point precision. For this reason, it is usually better to use the function linspace that receives as an argument the number of elements that we want, instead of the step:

# In[22]:


from numpy import pi
np.linspace( 0, 2, 9 ) # 9 numbers from 0 to 2


# In[23]:


x = np.linspace( -2*pi, 2*pi, 100 ) # useful to evaluate function at lots of points
f = np.sin(x)

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(x, f)
plt.xlabel('x')
plt.ylabel('y = $\sin(x) = - \int_a^b \cos(x)$')
plt.show()


# **See also
# array, zeros, zeros_like, ones, ones_like, empty, empty_like, arange, linspace, numpy.random.mtrand.RandomState.rand, numpy.random.mtrand.RandomState.randn, fromfunction, fromfile**

# ### Printing Arrays
# 
# When you print an array, NumPy displays it in a similar way to nested lists, but with the following layout:
# 
# * the last axis is printed from left to right,
# * the second-to-last is printed from top to bottom,
# * the rest are also printed from top to bottom, with each slice separated from the next by an empty line.
# 
# One-dimensional arrays are then printed as rows, bidimensionals as matrices and tridimensionals as lists of matrices.

# In[24]:


a = np.arange(6)                         # 1d array
print(a)


# In[25]:


b = np.arange(12).reshape(4,3)           # 2d array
print(b)


# In[26]:


c = np.arange(24).reshape(2,3,4)         # 3d array
print(c)


# If an array is too large to be printed, NumPy automatically skips the central part of the array and only prints the corners:

# In[27]:


print(np.arange(10000))


# In[28]:


print(np.arange(10000).reshape(100,100))


# To disable this behaviour and force NumPy to print the entire array, you can change the printing options using set_printoptions.

# In[29]:


import sys
np.set_printoptions(threshold=sys.maxsize)       # sys module should be imported


# In[30]:


print(np.arange(400).reshape(20,20))


# ### Basic Operations
# 
# Arithmetic operators on arrays apply elementwise. A new array is created and filled with the result.

# In[31]:


a = np.array( [20,30,40,50] )
b = np.arange( 4 )
b


# In[32]:


c = a-b
c


# In[33]:


b**2


# In[34]:


10*np.sin(a)


# In[35]:


a < 35


# Unlike in many matrix languages, the product operator * operates elementwise in NumPy arrays. The matrix product can be performed using the @ operator (in python >=3.5) or the dot function or method:

# In[36]:


A = np.array( [[1,1],
            [0,1]] )
B = np.array( [[2,0],
            [3,4]] )
A * B                       # elementwise product


# In[37]:


A @ B                       # matrix product


# In[38]:


A.dot(B)                    # another matrix product


# Some operations, such as += and *=, act in place to modify an existing array rather than create a new one.

# In[39]:


a = np.ones((2,3), dtype=int)
b = np.random.random((2,3))
a *= 3
a


# In[40]:


b += a
b


# In[41]:


# a += b                  # b is not automatically converted to integer type


# **Upcasting**
# 
# * When operating with arrays of different types, the type of the resulting array corresponds to the more general or precise one.

# In[42]:


a = np.ones(3, dtype=np.int32)
b = np.linspace(0,pi,3)
b.dtype.name


# In[43]:


c = a+b
c


# In[44]:


c.dtype.name


# In[45]:


d = np.exp(c*1j)
d


# In[46]:


d.dtype.name


# Many unary operations, such as computing the sum of all the elements in the array, are implemented as methods of the ndarray class.

# In[47]:


a = np.random.random((2,3))
a


# In[48]:


a.sum()


# In[49]:


a.min()


# In[50]:


a.max()


# By default, these operations apply to the array as though it were a list of numbers, regardless of its shape. However, by specifying the axis parameter you can apply an operation along the specified axis of an array:

# In[51]:


b = np.arange(12).reshape(3,4)
b


# In[52]:


b.sum(axis=0)                            # sum of each column


# In[53]:


b.min(axis=1)                            # min of each row


# In[54]:


b.cumsum(axis=1)                         # cumulative sum along each row


# ### Universal Functions
# 
# NumPy provides familiar mathematical functions such as sin, cos, and exp. In NumPy, these are called “universal functions”(ufunc). Within NumPy, these functions operate elementwise on an array, producing an array as output.

# In[55]:


B = np.arange(3)
B


# In[56]:


np.exp(B)


# In[57]:


np.sqrt(B)


# In[58]:


C = np.array([2., -1., 4.])
np.add(B, C)


# **See also
# all, any, apply_along_axis, argmax, argmin, argsort, average, bincount, ceil, clip, conj, corrcoef, cov, cross, cumprod, cumsum, diff, dot, floor, inner, inv, lexsort, max, maximum, mean, median, min, minimum, nonzero, outer, prod, re, round, sort, std, sum, trace, transpose, var, vdot, vectorize, where**

# ### Indexing, Slicing and Iterating
# 
# **One-dimensional** arrays can be indexed, sliced and iterated over, much like lists and other Python sequences.

# In[59]:


a = np.arange(10)**3
a


# In[60]:


a[2]


# In[61]:


a[2:5]


# In[62]:


a[:6:2] = -1000    # equivalent to a[0:6:2] = -1000; from start to position 6, exclusive, set every 2nd element to -1000
a


# In[63]:


a[ : :-1]                                 # reversed a


# In[64]:


# for i in a:                             # NumPy, RuntimeWarning: invalid value encountered in power
for i in a.astype('complex'):
    print(i**(1/3.0))


# **Multidimensional** arrays can have one index per axis. These indices are given in a tuple separated by commas:

# In[65]:


def f(x,y):
    return 10*x+y

b = np.fromfunction(f,(5,4),dtype=int)
b


# In[66]:


b[2,3]


# In[67]:


b[0:5, 1]                       # each row in the second column of b


# In[68]:


b[ : ,1]                        # equivalent to the previous example


# In[69]:


b[1:3, : ]                      # each column in the second and third row of b


# When fewer indices are provided than the number of axes, the missing indices are considered complete slices:

# In[70]:


b[-1]                                  # the last row. Equivalent to b[-1,:]


# The expression within brackets in b[i] is treated as an i followed by as many instances of : as needed to represent the remaining axes. NumPy also allows you to write this using dots as b[i,...].
# 
# The **dots** (...) represent as many colons as needed to produce a complete indexing tuple. For example, if x is an array with 5 axes, then
# 
# - x[1,2,...] is equivalent to x[1,2,:,:,:],
# - x[...,3] to x[:,:,:,:,3] and
# - x[4,...,5,:] to x[4,:,:,5,:].

# In[71]:


c = np.array( [[[  0,  1,  2],               # a 3D array (two stacked 2D arrays)
                [ 10, 12, 13]],
               [[100,101,102],
                [110,112,113]]])
print(c)


# In[72]:


c.shape


# In[73]:


c[1,...]                                   # same as c[1,:,:] or c[1]


# In[74]:


c[...,2]                                   # same as c[:,:,2]


# **Iterating** over multidimensional arrays is done with respect to the first axis:

# In[75]:


print(b)
for row in b:
    print(row)


# However, if one wants to perform an operation on each element in the array, one can use the flat attribute which is an iterator over all the elements of the array:

# In[76]:


for element in b.flat:
    print(element)


# **See also
# Indexing, Indexing (reference), newaxis, ndenumerate, indices**

# ## Shape Manipulation
# 
# ### Changing the shape of an array
# 
# An array has a shape given by the number of elements along each axis:

# In[77]:


a = np.floor(10*np.random.random((3,4)))
a


# In[78]:


a.shape


# The shape of an array can be changed with various commands. Note that the following three commands all return a modified array, but do not change the original array:

# In[79]:


a.ravel()  # returns the array, flattened


# In[80]:


a.reshape(6,2)  # returns the array with a modified shape


# In[81]:


a.T  # returns the array, transposed


# In[82]:


a.T.shape


# In[83]:


a.shape


# * The order of the elements in the array resulting from ravel() is normally “C-style”, that is, the rightmost index “changes the fastest”, so the element after a[0,0] is a[0,1].
# * If the array is reshaped to some other shape, again the array is treated as “C-style”.
# * NumPy normally creates arrays stored in this order, so ravel() will usually not need to copy its argument
# * **But** if the array was made by taking slices of another array or created with unusual options, it may need to be copied.
# * The functions ravel() and reshape() can also be instructed, using an optional argument, to use **FORTRAN-style** arrays, in which the leftmost index changes the fastest.

# * The **reshape** function returns its argument with a modified shape
# * The ndarray.resize method modifies the array itself:

# In[84]:


a


# In[85]:


a.resize((2,6))
a


# If a dimension is given as -1 in a reshaping operation, the other dimensions are automatically calculated:

# In[86]:


a.reshape(2,-1)


# **See also
# ndarray.shape, reshape, resize, ravel**

# ### Stacking together different arrays
# 
# Several arrays can be stacked together along different axes:

# In[87]:


a = np.floor(10*np.random.random((2,2)))
print(a)
b = np.floor(10*np.random.random((2,2)))
print(b)


# In[88]:


np.vstack((a,b))


# In[89]:


np.hstack((a,b))


# The function **column_stack** stacks 1D arrays as columns into a 2D array. It is equivalent to hstack only for 2D arrays:

# In[90]:


from numpy import newaxis
c = np.column_stack((a,b))     # with 2D arrays
print('\na:')
print(a)
print('\nb:')
print(b)
print('\nc = np.column_stac((a,b)):')
print(c)


# In[91]:


a = np.array([4.,2.])
b = np.array([3.,8.])
c = np.column_stack((a,b))     # returns a 2D array

print('\na:')
print(a)
print('\na.shape:')
print(a.shape)
print('\nb:')
print(b)
print('\nb.shape:')
print(b.shape)
print('\nc = np.column_stac((a,b)):')
print(c)
print('\nc.shape:')
print(c.shape)


# In[92]:


c = np.hstack((a,b))           # the result is different
print('\na:')
print(a)
print('\na.shape:')
print(a.shape)
print('\nb:')
print(b)
print('\nb.shape:')
print(b.shape)
print('\nc = np.column_stac((a,b)):')
print(c)
print('\nc.shape:')
print(c.shape)


# In[93]:


c = a[:,newaxis]               # this allows to have a 2D columns vector
print('\nc = a[:,newaxis]:')
print(c)
print('\nc.shape:')
print(c.shape)


# In[94]:


c = np.column_stack((a[:,newaxis],b[:,newaxis]))
c


# In[95]:


c = np.hstack((a[:,newaxis],b[:,newaxis]))   # the result is the same as column stack
c


# 1. On the other hand, the function **ma.row_stack** is equivalent to **vstack** for any input arrays.
# 
# 1. In general:
# 
# 
# * For arrays with more than two dimensions, **hstack** stacks along their **second** axes (columns).
# * **vstack** stacks along their **first** axes (rows).
# * Concatenate allows for an optional arguments giving the number of the axis along which the concatenation should happen.

# **Note**
# 
# In complex cases, r_ and c_ are useful for creating arrays by stacking numbers along one axis. They allow the use of range literals (“:”)
# 

# In[96]:


np.r_[1:4, 11:3:-2, -4]


# When used with arrays as arguments, r_ and c_ are similar to vstack and hstack in their default behavior, but allow for an optional argument giving the number of the axis along which to concatenate.

# **See also
# hstack, vstack, column_stack, concatenate, c_, r_**

# ### Splitting one array into several smaller ones
# 
# * **hsplit** -- split an array along its horizontal axis, either by specifying the number of equally shaped arrays to return, or by specifying the columns after which the division should occur:

# In[97]:


a = np.floor(10*np.random.random((2,12)))
a


# In[98]:


b = np.hsplit(a,3)   # Split a into 3
print('\na:')
print(a)
print("\nb = np.hsplit(a,3):")
for section in b:
    print(section)


# In[99]:


b = np.hsplit(a,2)   # Split a into 3
print('\na:')
print(a)
print("\nb = np.hsplit(a,3):")
for section in b:
    print(section)


# In[100]:


b = np.hsplit(a,(3,4))   # Split a after the third and the fourth column
print('\na:')
print(a)
print("\nb = np.hsplit(a,(3,4)):")
for section in b:
    print(section)


# * **vsplit** splits along the vertical axis
# * **array_split** allows one to specify along which axis to split.

# ## Copies and Views
# 
# * When operating and manipulating arrays, their data is sometimes copied into a new array and sometimes not.
# * This is often a source of confusion for beginners. There are three cases:

# ### No Copy at All
# 
# Simple assignments make no copy of array objects or of their data.

# In[101]:


a = np.arange(12)
b = a            # no new object is created
print('\na:')
print(a)
print('\nb:')
print(b)


# In[102]:


b is a           # a and b are two names for the same ndarray object


# In[103]:


b.shape = 3,4    # changes the shape of a
a.shape
print('\na:')
print(a)
print('\nb:')
print(b)


# Python passes mutable objects as references, so function calls make no copy.

# In[104]:


def f(x):
    print(id(x))

id(a)                           # id is a unique identifier of an object


# In[105]:


f(a)


# ### View or Shallow Copy
# 
# Different array objects can share the same data. The view method creates a new array object that looks at the same data.

# In[106]:


c = a.view()
c is a


# In[107]:


id(a)


# In[108]:


id(c)


# In[109]:


c.base is a                        # c is a view of the data owned by a


# In[110]:


c.flags.owndata


# In[111]:


c.shape = 2,6                      # a's shape doesn't change
print(a.shape)
print(c.shape)


# In[112]:


c[0,4] = 1234                      # but a's data changes
a


# Slicing an array returns a view of it:

# In[113]:


s = a[ : , 1:3]
s[:] = 10           # s[:] is a view of s. Note the difference between s=10 and s[:]=10
a


# ### Deep Copy
# 
# The copy method makes a complete copy of the array and its data.

# In[114]:


d = a.copy()                          # a new array object with new data is created
d is a


# In[115]:


id(a)


# In[116]:


id(c)


# In[117]:


d.base is a                           # d doesn't share anything with a


# In[118]:


d[0,0] = 9999                         # d doesn't share anything with a
a


# Sometimes copy should be called after slicing if the original array is not required anymore.
# 
# For example:
# 
# - If a is a huge intermediate result and the final result b only contains a small fraction of a, a deep copy should be made when constructing b with slicing:
# 
# 

# In[119]:


a = np.arange(int(1e8))
b = a[:100].copy()
del a  # the memory of ``a`` can be released.
print('\nb:')
print(b)
print('\nb.shape:')
print(b.shape)


# If b = a[:100] is used instead, a is referenced by b and will persist in memory even if del a is executed.

# ### Functions and Methods Overview¶
# 
# **Array Creation**
# 
# arange, array, copy, empty, empty_like, eye, fromfile, fromfunction, identity, linspace, logspace, mgrid, ogrid, ones, ones_like, r, zeros, zeros_like
# 
# **Conversions**
# ndarray.astype, atleast_1d, atleast_2d, atleast_3d, mat
# 
# **Manipulations**
# 
# array_split, column_stack, concatenate, diagonal, dsplit, dstack, hsplit, hstack, ndarray.item, newaxis, ravel, repeat, reshape, resize, squeeze, swapaxes, take, transpose, vsplit, vstack
# 
# **Questions**
# 
# all, any, nonzero, where

# **Ordering**
# 
# argmax, argmin, argsort, max, min, ptp, searchsorted, sort
# 
# **Operations**
# 
# choose, compress, cumprod, cumsum, inner, ndarray.fill, imag, prod, put, putmask, real, sum
# 
# **Basic Statistics**
# 
# cov, mean, std, var
# 
# **Basic Linear Algebra**
# 
# cross, dot, outer, linalg.svd, vdot

# ## Less Basic
# 
# ### Broadcasting rules
# 
# * Broadcasting allows universal functions to deal in a meaningful way with inputs that do not have exactly the same shape.
# 
# 1. The first rule of broadcasting is that if all input arrays do not have the same number of dimensions, a “1” will be repeatedly prepended to the shapes of the smaller arrays until all the arrays have the same number of dimensions.
# 
# 1. The second rule of broadcasting ensures that arrays with a size of 1 along a particular dimension act as if they had the size of the array with the largest shape along that dimension. The value of the array element is assumed to be the same along that dimension for the “broadcast” array.
# 
# After application of the broadcasting rules, the sizes of all arrays must match. More details can be found in Broadcasting.

# ## Fancy indexing and index tricks
# 
# * NumPy offers more indexing facilities than regular Python sequences.
# 
# * In addition to indexing by integers and slices, as we saw before, arrays can be indexed by:
# 
# 
# 1. arrays of integers, and;
# 1. arrays of booleans.

# ### Indexing with Arrays of Indices

# In[120]:


a = np.arange(12)**2                       # the first 12 square numbers
i = np.array( [ 1,1,3,8,5 ] )              # an array of indices
a[i]                                       # the elements of a at the positions i

print('\na:')
print(a)
print('\ni:')
print(i)
print('\na[i]:')
print(a[i])


# When the indexed array a is multidimensional, a single array of indices refers to the first dimension of a. The following example shows this behavior by converting an image of labels into a color image using a palette.

# In[121]:


palette = np.array( [ [0,0,0],                # black
                      [255,0,0],              # red
                      [0,255,0],              # green
                      [0,0,255],              # blue
                      [255,255,255] ] )       # white

image = np.array( [ [ 0, 1, 2, 0 ],           # each value corresponds to a color in the palette
                    [ 0, 3, 4, 0 ]  ] )
palette[image]                            # the (2,4,3) color image


# We can also give indexes for more than one dimension. The arrays of indices for each dimension must have the same shape.

# In[122]:


a = np.arange(12).reshape(3,4)
print('\na:')
print(a)


# In[123]:


i = np.array( [ [0,1],                        # indices for the first dim of a
                [1,2] ] )
j = np.array( [ [2,1],                        # indices for the second dim
                [3,3] ] )
print('\ni:')
print(i)
print('\nj:')
print(j)


# In[124]:


print('\na:')
print(a)
print('\ni:')
print(i)
print('\nj:')
print(j)
b = a[i,j]                                     # i and j must have equal shape
print('\nb = a[i,j]:')
print(b)


# In[125]:


print('\na:')
print(a)
print('\ni:')
print(i)
print('\nj:')
print(j)
b = a[i,2]
print('\nb = a[i,2]:')
print(b)


# In[126]:


print('\na:')
print(a)
print('\ni:')
print(i)
print('\nj:')
print(j)
b = a[:,j]                                     # i and j must have equal shape
                                               # ':' = [0,1,2]
print('\nb = a[:,j]:')
print(b)


# Naturally, we can put i and j in a sequence (say a list) and then do the indexing with the list.

# In[127]:


print('\na:')
print(a)
print('\ni:')
print(i)
print('\nj:')
print(j)
l = (i,j)
print('\nl:')
print(l)
b = a[l]                                       # equivalent to a[i,j]
print('\nb = a[l]:')
print(b)


# However, we **can not** do this by putting i and j into an array, because this array will be interpreted as indexing the first dimension of a.

# In[128]:


# s = np.array( [i,j] )
# a[s]


# In[129]:


# a[tuple(s)]                                # same as a[i,j]


# Another common use of indexing with arrays is the search of the maximum value of time-dependent series:

# In[130]:


time = np.linspace(20, 145, 5)                 # time scale
data = np.sin(np.arange(20)).reshape(5,4)      # 4 time-dependent series
time


# In[131]:


data


# In[132]:


import matplotlib.pyplot as plt
plt.plot(time,data)
plt.legend(['0','1','2','3'])
plt.show()


# In[133]:


ind = data.argmax(axis=0)                  # index of the maxima for each series
ind


# In[134]:


time_max = time[ind]                       # times corresponding to the maxima
time_max


# In[135]:


data_max = data[ind, range(data.shape[1])] # => data[ind[0],0], data[ind[1],1]...


# In[136]:


np.all(data_max == data.max(axis=0))


# In[137]:


import matplotlib.pyplot as plt
plt.plot(time,data)
plt.plot(time_max, data[(ind, np.arange(data.shape[1]))], 'ro')
plt.legend(['0','1','2','3'])
plt.show()


# You can also use indexing with arrays as a target to assign to:

# In[138]:


a = np.arange(5)
a[[1,3,4]] = 0
a


# However, when the list of indices contains repetitions, the assignment is done several times, leaving behind the last value:

# In[139]:


a = np.arange(5)
a[[0,0,2]]=[1,2,3]
a


# This is reasonable enough, but watch out if you want to use Python’s += construct, as it may not do what you expect:

# In[140]:


a = np.arange(5)
print('\na:')
print(a)
a[[0,0,2]]+=1
a
print('\na:')
print(a)


# Even though 0 occurs twice in the list of indices, the 0th element is only incremented once. This is because Python requires “a+=1” to be equivalent to “a = a + 1”.

# ### Indexing with Boolean Arrays
# 
# * When we index arrays with arrays of (integer) indices we are providing the list of indices to pick.
# * With boolean indices the approach is different; we explicitly choose which items in the array we want and which ones we don’t.
# * The most natural way one can think of for boolean indexing is to use boolean arrays that have the same shape as the original array:

# In[141]:


a = np.arange(12).reshape(3,4)
print('\na:')
print(a)
b = a > 4
b                                          # b is a boolean with a's shape
print('\nb = a > 4:')
print(b)


# In[142]:


a[b]                                       # 1d array with the selected elements


# This property can be very useful in assignments:

# In[143]:


print('\na:')
print(a)
a[b] = 0                                   # All elements of 'a' higher than 4 become 0
a
print('\na[a > 4] = 0:')
print(a)


# You can look at the following example to see how to use boolean indexing to generate an image of the Mandelbrot set:

# In[144]:


import numpy as np
import matplotlib.pyplot as plt
def mandelbrot( h,w, maxit=20 ):
    """Returns an image of the Mandelbrot fractal of size (h,w)."""
    y,x = np.ogrid[ -1.4:1.4:h*1j, -2:0.8:w*1j ]
    c = x+y*1j
    z = c
    divtime = maxit + np.zeros(z.shape, dtype=int)

    for i in range(maxit):
        z = z**2 + c
        diverge = z*np.conj(z) > 2**2            # who is diverging
        div_now = diverge & (divtime==maxit)  # who is diverging now
        divtime[div_now] = i                  # note when
        z[diverge] = 2                        # avoid diverging too much

    return divtime

plt.imshow(mandelbrot(400,400))
plt.show()


# The second way of indexing with booleans is more similar to integer indexing; for each dimension of the array we give a 1D boolean array selecting the slices we want:

# In[145]:


a = np.arange(12).reshape(3,4)
b1 = np.array([False,True,True])             # first dim selection
b2 = np.array([True,False,True,False])       # second dim selection
a[b1,:]                                   # selecting rows


# In[146]:


a[b1]                                     # same thing


# In[147]:


a[:,b2]                                   # selecting columns


# In[148]:


a[b1,b2]                                  # a weird thing to do


# Note that the length of the 1D boolean array must coincide with the length of the dimension (or axis) you want to slice. In the previous example, b1 has length 3 (the number of rows in a), and b2 (of length 4) is suitable to index the 2nd axis (columns) of a.

# ### The ix_() function
# 
# * The ix_ function can be used to combine different vectors so as to obtain the result for each n-uplet.
# * For example, if you want to compute all the a+b*c for all the triplets taken from each of the vectors a, b and c:

# In[149]:


a = np.array([2,3,4,5])
a


# In[150]:


b = np.array([8,5,4])
b


# In[151]:


c = np.array([5,4,6,8,3])
c


# In[152]:


ax,bx,cx = np.ix_(a,b,c)
ax


# In[153]:


bx


# In[154]:


cx


# In[155]:


ax.shape, bx.shape, cx.shape


# In[156]:


result = ax+bx*cx
result


# In[157]:


result[3,2,4]


# In[158]:


a[3]+b[2]*c[4]


# You could also implement the reduce as follows:

# In[159]:


def ufunc_reduce(ufct, *vectors):
   vs = np.ix_(*vectors)
   r = ufct.identity
   for v in vs:
       r = ufct(r,v)
   return r

ufunc_reduce(np.add,a,b,c)


# The advantage of this version of reduce compared to the normal ufunc.reduce is that it makes use of the Broadcasting Rules in order to avoid creating an argument array the size of the output times the number of vectors.

# ### Indexing with strings
# 
# See Structured arrays.

# ## Linear Algebra

# In[160]:


import numpy as np
a = np.array([[1.0, 2.0], [3.0, 4.0]])
print(a)
b = a.transpose()
print(b)


# In[161]:


np.linalg.inv(a)


# In[162]:


u = np.eye(2) # unit 2x2 matrix; "eye" represents "I"
u


# In[163]:


j = np.array([[0.0, -1.0], [1.0, 0.0]])
j @ j        # matrix product


# In[164]:


np.trace(u)  # trace


# In[165]:


y = np.array([[5.], [7.]])
np.linalg.solve(a, y)


# In[166]:


np.linalg.eig(j)


# ## Tricks and Tips
# 
# ### “Automatic” Reshaping
# 
# To change the dimensions of an array, you can omit one of the sizes which will then be deduced automatically:

# In[167]:


a = np.arange(30)
a.shape = 2,-1,3  # -1 means "whatever is needed"
a.shape


# In[168]:


a


# ### Vector Stacking
# 
# How do we construct a 2D array from a list of equally-sized row vectors? In MATLAB this is quite easy: if x and y are two vectors of the same length you only need do m=[x;y]. In NumPy this works via the functions column_stack, dstack, hstack and vstack, depending on the dimension in which the stacking is to be done. For example:

# In[169]:


x = np.arange(0,10,2)                     # x=([0,2,4,6,8])
y = np.arange(5)                          # y=([0,1,2,3,4])
m = np.vstack([x,y])                      # m=([[0,2,4,6,8],
                                          #     [0,1,2,3,4]])
m


# In[170]:


xy = np.hstack([x,y])                     # xy =([0,2,4,6,8,0,1,2,3,4])
xy


# ### Histograms

# The NumPy histogram function applied to an array returns a pair of vectors: the histogram of the array and the vector of bins. Beware: matplotlib also has a function to build histograms (called hist, as in Matlab) that differs from the one in NumPy. The main difference is that pylab.hist plots the histogram automatically, while numpy.histogram only generates the data.

# In[171]:


import numpy as np
import matplotlib.pyplot as plt
# Build a vector of 10000 normal deviates with variance 0.5^2 and mean 2
mu, sigma = 2, 0.5
v = np.random.normal(mu,sigma,10000)
# Plot a normalized histogram with 50 bins
plt.hist(v, bins=50, density=1)       # matplotlib version (plot)
plt.show()


# In[172]:


# Compute the histogram with numpy and then plot it
(n, bins) = np.histogram(v, bins=50, density=True)  # NumPy version (no plot)
plt.plot(.5*(bins[1:]+bins[:-1]), n)
plt.show()

