import numpy as np
class c_array(np.ndarray): pass


# (
    # shape:    _ShapeLike,             tuple of ints (Shape of created array.)
    # dtype:    DTypeLike = ...,        data-type, optional (Any object that can be interpreted as a numpy data type.)
    # buffer:   _BufferType = ...,      object exposing buffer interface, optional (Used to fill the array with data.)
    # offset:   int = ...,              int, optional (Offset of array data in buffer.)
    # strides:  _ShapeLike = ...,       tuple of ints, optional (Strides of data in memory.)
    # order:    _OrderKACF = ...        optional (Row-major (C-style) or column-major (Fortran-style) order.)
    # ) -> c_array
    
    