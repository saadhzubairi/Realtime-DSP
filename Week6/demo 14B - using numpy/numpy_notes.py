
import numpy as np

# CONVERT LIST TO NUMPY N-DIMENSIONAL ARRAY

a = [1, 2, 3]

type(a)

b = np.array(a)

type(b)

b.dtype


# CONVERT LIST OR ARRAY OF FLOATS TO NUMPY N-DIMENSIONAL ARRAY

a = [1.1, 2.2, 8.8, 9.9]

b = np.array(a)

# Use type to get the data type.  Also use b.dtype to get the data type of the elements of a numpy array

type(b)			# numpy.ndarray
type(b[0])		# numpy.float64
b.dtype			# float64 <- FLOAT

# or

c = b.astype('int')

type(c)			# numpy.ndarray
type(c[0])		# numpy.int64
c.dtype			# int64 <- INTEGER

# or

d = np.asarray(b, dtype = 'int')

type(d)			# numpy.ndarray
type(d[0])		# numpy.int64
d.dtype			# int64 <- INTEGER

# or directly convert the list to a numpy array of integer data type

e = np.asarray(a, dtype = 'int')

type(e)			# <class 'numpy.ndarray'>
type(e[0])		# <class 'numpy.int64'>
e.dtype			# dtype('int64')

# * * * convert to 16 bit integer * * * 

c = b.astype('int16')

type(c)			# numpy.ndarray
type(c[0])		# numpy.int16
c.dtype			# int16 <- INTEGER

# or

d = np.asarray(b, dtype = 'int16')

type(d)			# numpy.ndarray
type(d[0])		# numpy.int16
d.dtype			# int16 <- INTEGER

# or directly convert the list to a numpy array of integer data type

e = np.asarray(a, dtype = 'int16')

type(e)			# <class 'numpy.ndarray'>
type(e[0])		# <class 'numpy.int16'>
e.dtype			# dtype('int16')


# CREATE SEQUENCE 0 to N

n = np.arange(10)   # numpy array 0, 1, 2, ..., 9


