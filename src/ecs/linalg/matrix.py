from common import mean, multiply, sqrt, square
from linalg.common import mean, minus, sqrt, square

from .common import dim, shape, zeros


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
                    raise Exception
                fa = -A[i][j]/A[j][j]
                _row_assign(A,i,j,fa)
                _row_assign(L,i,j,fa)
    #upper triangle    
    for j in range(N)[::-1]:
        for i in range(N)[::-1]:
            if i<j:
                if A[j][j]==0:
                    raise Exception
                fa = -A[i][j]/A[j][j]
                _row_assign(A,i,j,fa)
                _row_assign(L,i,j,fa)
                
    for i in range(len(L)):
        L[i] = [x / A[i][i] for x in L[i]]
    return matrix_matmul(R,L)
    
def matrix_transpose(A):
    assert(dim(A)==2)
    # m = shape(A)[0]
    # n = shape(A)[1]
    # result = []
    # for j in range(n):
    #     r = []
    #     for i in range(m):
    #         print(j,len(A[i]))
    #         r.append(A[i][j])
    #     result.append(r)
    result = [list(i) for i in zip(*A)]
    return result

def matrix_copy(A):
    assert(dim(A)==2)
    m,n = shape(A)
    R = zeros((m,n))
    for i in range(m):
        for j in range(n):
            R[i][j] = A[i][j]
    return R



def vstack(list_of_matrix):
    R = []
    for m in list_of_matrix:
        R.extend(m)
    return R


def hstack(list_of_matrix):
    # from copy import deepcopy
    # list_of_matrix = deepcopy(list_of_matrix)
    assert(type(list_of_matrix)==list and len(list_of_matrix)>0)
    high = shape(list_of_matrix[0])[0]
    stacking_length = []

    # add @2018-04-11
    for i in range(len(list_of_matrix)):
        if dim(list_of_matrix[i])==1:
            list_of_matrix[i] = [[x] for x in list_of_matrix[i]]

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
def shift(A,shift_step,fill=None):
    assert(dim(A)==2)
    R = zeros(shape(A))
    for i in range(shape(A)[0]):
        for j in range(shape(A)[1]):
            if shift_step>=0:
                if i>=shift_step:
                    R[i][j] = A[i-shift_step][j]
                else:
                    if type(fill)==list:
                        R[i][j] = fill[j]
                    else:
                        R[i][j] = fill
            else:
                if (i-shift_step)<shape(A)[0]:
                    R[i][j] = A[i-shift_step][j]
                else:
                    if type(fill)==list:
                        R[i][j] = fill[j]
                    else:
                        R[i][j] = fill
    return R


# column vector corrcoef
def corrcoef(A):
    assert(dim(A)==2)
    m,n = shape(A)
    def _corr(A,i,j):
        assert(dim(A)==2)
        m,n = shape(A)
        A_T = matrix_transpose(A)
        
        X,Y = A_T[i],A_T[j] # X,Y = col(A,i),col(A,j)

        mean_X,mean_Y = mean(X),mean(Y)
        X_ = [k-mean_X for k in X]
        Y_ = [k-mean_Y for k in Y]
        numerator = mean(multiply(X_,Y_))
        # print(sqrt(mean(square(X_))))

        denominator = sqrt(mean(square(X_)))*sqrt(mean(square(Y_)))
        if denominator == 0:
            return 0
        else:
            r = (numerator)/(denominator)
            return r

    R = zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i==j:
                R[i][j] = 1
            elif i>j:
                R[i][j] = R[j][i]
            else:
                R[i][j] = _corr(A,i,j)
    return R


def stdev(X,axis=1):
    assert(dim(X)==2)
    assert(axis==1)
    X_T = matrix_transpose(X)
    m = mean(X,axis=1)
    R = []
    for j in range(shape(X)[1]):
        R.append(sqrt(mean(square(minus(X_T[j],m[j])))))
    return R

# add @ 2018-04-11
def diag(A):
    assert(dim(A)==1)
    R = zeros((shape(A)[0],shape(A)[0]))
    for i in range(len(A)):
        R[i][i] = A[i]
    return R