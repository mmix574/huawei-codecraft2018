from .common import dim,shape
from .common import zeros
# moved to common 

# def zeros(*args,**kwargs):
#     # fix @ 2018-03-20
#     if 'dtype' in kwargs:
#         dtype = kwargs['dtype']
#     else:
#         dtype = float
#     assert(len(args)>0)
#     s = tuple(args[0]) if type(args[0])==tuple else tuple(args)

#     if type(s)==int:
#         return [dtype(0) for _ in range(s)]
#     elif type(s) == tuple and len(s)==1:
#         return [dtype(0) for _ in range(s[0])]
#     else:
#         import copy
#         r = zeros(s[1:],dtype=dtype)
#         R = []
#         for i in range(s[0]):
#             R.append(copy.deepcopy(r)) # fix a bug here
#     return R

# def ones(*args,**kwargs):
#     if 'dtype' in kwargs:
#         dtype = kwargs['dtype']
#     else:
#         dtype = float
#     assert(len(args)>0)
#     assert(len(args)>0)
#     s = tuple(args[0]) if type(args[0])==tuple else tuple(args)

#     if type(s)==int:
#         return [dtype(1) for _ in range(s)]
#     elif type(s) == tuple and len(s)==1:
#         return [dtype(1) for _ in range(s[0])]
#     else:
#         import copy
#         r = ones(s[1:],dtype=dtype)
#         R = []
#         for i in range(s[0]):
#             R.append(copy.deepcopy(r))
#     return R

# freeze @2018-03-19
# use matrix_transpose(A)[i] instead

# columns slicing of matrix A
# def col(A,i):
#     assert(dim(A)==2)
#     m,n = shape(A)
#     return [A[k][i] for k in range(m)]

def matrix_matmul(A,B):
    assert(dim(A)==2 and dim(B)==2 and shape(A)[1]==shape(B)[0])
    def __sub_product(A,i,B,j):
        N = len(A[i])
        partial_sum = 0
        for k in range(N):
            partial_sum += A[i][k]*B[k][j]
        return partial_sum
    
    m = shape(A)[0]
    n = shape(B)[1]
    
    R = []
    for i in range(m):
        r = []
        for j in range(n):
            r.append(__sub_product(A,i,B,j))
        R.append(r)
    return R

# merged  @2018-03-18
def identity_matrix(s,dtype=float):
    assert(type(s)==int)
    result = []
    for i in range(s):
        r = []
        for j in range(s):
            if i==j:
                r.append(dtype(1))
            else:
                r.append(dtype(0))
        result.append(r)
    return result


def matrix_inverse(A):
    assert(dim(A)==2)
    N = shape(A)[0]
    L = identity_matrix(N)
    R = identity_matrix(N)

    def _row_assign(A,dest,source,factor):
        assert(dim(A)==2)
        A[dest] = [factor*A[source][i] +A[dest][i] for i in range(len(A[source]))]

    def _row_switch(A,dest,source):
        assert(dim(A)==2)
        t = A[dest]
        A[dest] = A[source]
        A[source] = t

    def _col_switch(A,dest,source):
        assert(dim(A)==2)
        m,n = shape(A)
        for i in range(m):
            t = A[i][dest]
            A[i][dest] = A[i][source]
            A[i][source] = t

    #down triangle     
    for j in range(N):
        for i in range(N):
            # select biggest element             
            if i==j:
                max_k = i
                max_w = j
                for k in range(i,N):
                    for w in range(j,N):
                        if A[k][w]>A[max_k][max_w]:
                            max_k,max_w = k,w
                _row_switch(A,i,max_k)
                _row_switch(L,i,max_k)
                _col_switch(A,j,max_w)
                _col_switch(R,j,max_w)
            if i>j:
                if A[j][j]==0:
                    return None
                fa = -A[i][j]/A[j][j]
                _row_assign(A,i,j,fa)
                _row_assign(L,i,j,fa)
    #upper triangle    
    for j in range(N)[::-1]:
        for i in range(N)[::-1]:
            if i<j:
                if A[j][j]==0:
                    return None
                fa = -A[i][j]/A[j][j]
                _row_assign(A,i,j,fa)
                _row_assign(L,i,j,fa)
                
    for i in range(len(L)):
        L[i] = [x / A[i][i] for x in L[i]]
    return matrix_matmul(R,L)
    
def matrix_transpose(A):
    assert(dim(A)==2)
    m = shape(A)[0]
    n = shape(A)[1]
    result = []
    for j in range(n):
        r = []
        for i in range(m):
            r.append(A[i][j])
        result.append(r)
    return result

# freeze @ 2018-03-20
# (use common.minus,comon.plus,commin.multiply instead.)

# def matrix_minus(A,B):
#     assert(dim(A)==2 and dim(B)==2)
#     assert(shape(A)==shape(B))
#     m,n = shape(A)
#     R = zeros((m,n))
#     for i in range(m):
#         for j in range(n):
#             R[i][j] = A[i][j] - B[i][j]
#     return R

# def matrix_plus(A,B):
#     assert(dim(A)==2 and dim(B)==2)
#     assert(shape(A)==shape(B))
#     m,n = shape(A)
#     R = zeros((m,n))
#     for i in range(m):
#         for j in range(n):
#             R[i][j] = A[i][j] + B[i][j]
#     return R

# def matrix_multiply(A,B):
#     assert(dim(A)==2 and dim(B)==2)
#     assert(shape(A)==shape(B))
#     m,n = shape(A)
#     R = zeros((m,n))
#     for i in range(m):
#         for j in range(n):
#             R[i][j] = A[i][j] * B[i][j]
#     return R

# def matrix_multiply_factor(A,fa):
#     assert(dim(A)==2 and (type(fa)==float or type(fa)==int))
#     m,n = shape(A)
#     R = zeros((m,n))
#     for i in range(m):
#         for j in range(n):
#             R[i][j] = A[i][j] * fa
#     return R

def matrix_copy(A):
    assert(dim(A)==2)
    m,n = shape(A)
    R = zeros((m,n))
    for i in range(m):
        for j in range(n):
            R[i][j] = A[i][j]
    return R


def vstack(list_of_matrix):
    assert(type(list_of_matrix)==list and len(list_of_matrix)>0)
    width = shape(list_of_matrix[0])[1]
    stacking_length = []
    for i in range(len(list_of_matrix)):
        assert(dim(list_of_matrix[i])==2)
        assert(shape(list_of_matrix[i])[1]==width)
        stacking_length.append(shape(list_of_matrix[i])[0])

    R = zeros(sum(stacking_length),width)
    for i in range(len(list_of_matrix)):
        m,n = shape(list_of_matrix[i])
        start = sum(stacking_length[:i])
        # element wise copy
        for j in range(m):
            for k in range(n):
                R[j+start][k] = list_of_matrix[i][j][k]
    return R


def hstack(list_of_matrix):
    assert(type(list_of_matrix)==list and len(list_of_matrix)>0)
    high = shape(list_of_matrix[0])[0]
    stacking_length = []
    for i in range(len(list_of_matrix)):
        assert(dim(list_of_matrix[i])==2)
        assert(shape(list_of_matrix[i])[0]==high)
        stacking_length.append(shape(list_of_matrix[i])[1]) 

    R = zeros(high,sum(stacking_length))
    for i in range(len(list_of_matrix)):
        m,n = shape(list_of_matrix[i])
        start = sum(stacking_length[:i])
        # element wise copy
        for j in range(m):
            for k in range(n):
                R[j][k+start] = list_of_matrix[i][j][k]
    return R

def matrix_reduce_sum(A,axis=None):
    assert(dim(A)==2)    
    if axis!=None:
        assert(dim(A)>=axis)
    A_T = matrix_transpose(A)

    if axis==0:
        return [sum(col) for col in A_T]
    elif axis==1:
        return [sum(row) for row in A]
    else:
        m,n = shape(A)
        count = 0
        for i in range(m):
            for j in range(n):
                count+=A[i][j]
        return count

def matrix_reduce_mean(A,axis=None):
    assert(dim(A)==2)    
    if axis!=None:
        assert(dim(A)>=axis)
    A_T = matrix_transpose(A)
    m,n = shape(A)

    if axis==0:
        return [sum(col)/float(m) for col in A_T]
    elif axis==1:
        return [sum(row)/float(n) for row in A]
    else:
        count = 0
        for i in range(m):
            for j in range(n):
                count+=A[i][j]
        return count/float(m*n)

# add @2018-03-20
def shift(A,shift_step):
    assert(dim(A)==2)
    R = zeros(shape(A))
    for i in range(shape(A)[0]):
        for j in range(shape(A)[1]):
            if shift_step>=0:
                if i>=shift_step:
                    R[i][j] = A[i-shift_step][j]
                else:
                    R[i][j] = None
            else:
                if (i-shift_step)<shape(A)[0]:
                    R[i][j] = A[i-shift_step][j]
                else:
                    R[i][j] = None
    return R

def load(path):
    import os
    assert(os.path.exists(path))
    
