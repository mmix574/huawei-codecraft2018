# common fucntion for both vector and matrix 
# broadcasting supported
# scalar supported

def dim(A):
    assert(type(A)==list)
    d = 0
    B = A
    while(type(B)==list and len(B)!=0):
        d += 1
        B = B[0]
    return d
# return the shape of a matrix or a tensor 

def shape(A):
    assert(type(A)==list)
    B = A
    s = []
    while(type(B)==list and len(B)!=0):
        s.append(len(B))
        B = B[0]
    return tuple(s)

# add @2018-03-18
def reshape(A,dimension):
    assert(type(A)==list)
    if type(dimension)==int:
        dimension = (dimension,)
    assert(type(dimension)==tuple or type(dimension)==list)

    # element counts
    def _count(A,e):
        count = 0
        for i in range(len(A)):
            if A[i]==e:
                count+=1
        return count

    assert(_count(dimension,-1)<=1)

    # volume calculate
    def _volume(d):
        product = 1
        for i in range(len(d)):
            if d[i]>0:
                product *= d[i]
        return product
    assert(_volume(shape(A))%_volume(dimension)==0)


    # print(dimension)
    # if -1 in dimension:
    #     index= 0 
    #     for i in range(len(dimension)):
    #         if dimension[i]==-1:
    #             index = i
    #             break
    #     _volume(dimension)
    #     # dimension[index] = _volume(shape(A))/_volume(dimension)

        
    # flatten reshape     
    def __reshape(A,dimension):
        if len(dimension)==1:
            return A
        else:
            dimension = list(dimension)
            if -1 in dimension:
                index= 0 
                for i in range(len(dimension)):
                    if dimension[i]<0:
                        index = i
                        break
                dimension[index] = _volume(shape(A))/_volume(dimension)
            R = []
            shape_A = shape(A)
            count = 1
            for i in range(len(shape_A)):
                count *= shape_A[i]
            for i in range(dimension[0]):
                R.append(A[i*(count//dimension[0]):(i+1)*(count//dimension[0])])
            # fix a bug
            if len(dimension)>1:
                for i in range(dimension[0]):
                    R[i] = __reshape(R[i],dimension[1:])
            return R
    B = flatten(A)
    return __reshape(B,dimension)

# broadcasting supportted
def plus(A,B):
    assert(type(A)==list)
    if dim(A)==1 and type(B)==list:
        assert(shape(A)==shape(B))
        return [A[i]+B[i] for i in range(len(A))]

    elif dim(A)==1 and (type(B)==float or type(B)==int):
        return [x+B for x in A]

    elif dim(A)==2 and type(B)==list:
        assert(shape(A)==shape(B))
        R = []
        for i in range(shape(A)[0]):
            r = []
            for j in range(shape(A)[1]):
                r.append(A[i][j]+B[i][j])
            R.append(r)
        return R
    elif dim(A)==2 and (type(B)==float or type(B)==int):
        R = []
        for i in range(shape(A)[0]):
            R.append([x+B for x in A[i]])
        return R
    return None


def minus(A,B):
    assert(type(A)==list)
    if dim(A)==1 and type(B)==list:
        assert(shape(A)==shape(B))
        return [A[i]-B[i] for i in range(len(A))]

    elif dim(A)==1 and (type(B)==float or type(B)==int):
        return [x-B for x in A]

    elif dim(A)==2 and type(B)==list:
        assert(shape(A)==shape(B))
        R = []
        for i in range(shape(A)[0]):
            r = []
            for j in range(shape(A)[1]):
                r.append(A[i][j]-B[i][j])
            R.append(r)
        return R
    elif dim(A)==2 and (type(B)==float or type(B)==int):
        R = []
        for i in range(shape(A)[0]):
            R.append([x-B for x in A[i]])
        return R
    return None

def multiply(A,B):
    assert(type(A)==list)
    if dim(A)==1 and type(B)==list:
        assert(shape(A)==shape(B))
        return [A[i]*B[i] for i in range(len(A))]

    elif dim(A)==1 and (type(B)==float or type(B)==int):
        return [x*B for x in A]

    elif dim(A)==2 and type(B)==list:
        assert(shape(A)==shape(B))
        R = []
        for i in range(shape(A)[0]):
            r = []
            for j in range(shape(A)[1]):
                r.append(A[i][j]*B[i][j])
            R.append(r)
        return R
    elif dim(A)==2 and (type(B)==float or type(B)==int):
        R = []
        for i in range(shape(A)[0]):
            R.append([x*B for x in A[i]])
        return R
    return None

def zeros(*args,**kwargs):
    # fix @ 2018-03-20
    if 'dtype' in kwargs:
        dtype = kwargs['dtype']
    else:
        dtype = float
    assert(len(args)>0)
    s = tuple(args[0]) if type(args[0])==tuple else tuple(args)

    if type(s)==int:
        return [dtype(0) for _ in range(s)]
    elif type(s) == tuple and len(s)==1:
        return [dtype(0) for _ in range(s[0])]
    else:
        import copy
        r = zeros(s[1:],dtype=dtype)
        R = []
        for i in range(s[0]):
            R.append(copy.deepcopy(r)) # fix a bug here
    return R

def ones(*args,**kwargs):
    if 'dtype' in kwargs:
        dtype = kwargs['dtype']
    else:
        dtype = float
    assert(len(args)>0)
    assert(len(args)>0)
    s = tuple(args[0]) if type(args[0])==tuple else tuple(args)

    if type(s)==int:
        return [dtype(1) for _ in range(s)]
    elif type(s) == tuple and len(s)==1:
        return [dtype(1) for _ in range(s[0])]
    else:
        import copy
        r = ones(s[1:],dtype=dtype)
        R = []
        for i in range(s[0]):
            R.append(copy.deepcopy(r))
    return R

def dot(A,B):
    assert(type(A)==list and type(B)==list)
    assert(dim(B)==1)
    if dim(A)==1:
        return sum([A[i]*B[i] for i in range(len(A))])
    elif dim(A)==2:
        assert(shape(A))
        R = []
        for k in range(shape(A)[0]):
            R.append(sum([A[k][i]*B[i] for i in range(len(A[k]))]))
        return R
    return None


# broadcasting funciton
def sqrt(A):
    # fix 
    import math
    if type(A)==int or type(A)==float:
        return math.pow(A,0.5)
    
    assert(type(A)==list)
    assert(dim(A)==1 or dim(A)==2)
    if dim(A)==1:
        return [math.pow(x,0.5) for x in A]
    elif dim(A)==2:
        R = []
        for k in range(shape(A)[0]):
            R.append([math.pow(x,0.5) for x in A[k]])
        return R
    return None

def square(A):
    # fix 
    import math
    if type(A)==int or type(A)==float:
        return math.pow(A,2)

    assert(type(A)==list)
    assert(dim(A)==1 or dim(A)==2)
    import math
    if dim(A)==1:
        return [math.pow(x,2) for x in A]
    elif dim(A)==2:
        R = []
        for k in range(shape(A)[0]):
            R.append([math.pow(x,2) for x in A[k]])
        return R
    return None


def abs(A):
    if type(A)==int or type(A)==float:
        return A if A>=0 else -A
    assert(type(A)==list)
    assert(dim(A)==1 or dim(A)==2)
    import math
    if dim(A)==1:
        return [x if x>=0 else -x for x in A]
    elif dim(A)==2:
        R = []
        for k in range(shape(A)[0]):
            R.append([x if x>=0 else -x for x in A[k]])
        return R
    return None

def sum(A,axis=None):
    assert(dim(A)==1 or dim(A)==2)
    if axis:
        assert(dim(A)>axis)
    if dim(A)==1:
        count = 0
        for i in range(len(A)):
            count+=A[i]
        return count
    elif dim(A)==2:
        if axis==None:
            count = 0
            for i in range(shape(A)[0]):
                for j in range(shape(A)[1]):
                    count+=A[i][j]
            return count
        elif axis==0:
            count = []
            for j in range(shape(A)[1]):
                c = 0
                for i in range(shape(A)[0]):
                    c+=A[i][j]
                count.append(c)
            return count
        elif axis==1:
            count = []
            for i in range(shape(A)[0]):
                c = 0
                for j in range(shape(A)[1]):
                    c+=A[i][j]
                count.append(c)
            return count
    return None

def mean(A,axis=None):
    assert(dim(A)==1 or dim(A)==2)
    if axis:
        assert(dim(A)>axis)
    if dim(A)==1:
        count = 0
        for i in range(len(A)):
            count+=A[i]
        return count/float(len(A))
    elif dim(A)==2:
        if axis==None:
            count = 0
            for i in range(shape(A)[0]):
                for j in range(shape(A)[1]):
                    count+=A[i][j]
            return count/float(shape(A)[0]*shape(A)[1])

        # fix a bug 
        elif axis==1:
            count = []
            for j in range(shape(A)[1]):
                c = 0
                for i in range(shape(A)[0]):
                    c+=A[i][j]
                c/=float(shape(A)[0])
                count.append(c)
            return count
            
         # fix a bug 
        elif axis==0:
            count = []
            for i in range(shape(A)[0]):
                c = 0
                for j in range(shape(A)[1]):
                    c+=A[i][j]
                c/=float(shape(A)[1])
                count.append(c)
            return count
    return None

# add @ 2018-03-28
def flatten(A):
    if dim(A)==1:
        return A
    s = shape(A)
    R = []
    for i in range(s[0]):
        R.extend(flatten(A[i]))
    return R

# add 2018-04-04
def fancy(*argv):
    A = argv[0]
    assert(type(A)==list)
    assert(dim(A)==len(argv)-1)

    if dim(A)==1:
        if type(argv[1])==int:
            if argv[1]==-1:
                return A
            else:
                return A[argv[1]]
        elif type(argv[1])==tuple or type(argv[1])==list:
            return [A[i] for i in range(argv[1][0],argv[1][1])]
        else:
            raise Exception            
    else:
        if type(argv[1])==int:
            if argv[1]==-1:
                return [fancy(A[i],*argv[2:]) for i in range(shape(A)[0])]
            else:
                return fancy(A[argv[1]],*argv[2:])
        elif type(argv[1]==tuple or type(argv[1]==list)):
            return [fancy(A[i],*argv[2:]) for i in range(argv[1][0],argv[1][1])]


