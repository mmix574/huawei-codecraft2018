from linalg.common import sqrt,mean,square,minus,mean,dim

# vector and matrix supportted
def l2_loss(y,y_):
    assert(dim(y)<=2 and dim(y_)<=2)
    # if dim(y) == 2:
    #     return mean(sqrt(mean(square(minus(y,y_)),axis=0)))
    # else:
    #     return sqrt(mean(square(minus(y,y_))))
    def _score_calc(y,y_):
        numerator = sqrt(mean(square(minus(y,y_))))
        return numerator

    if dim(y) == 1:
        return _score_calc(y,y_)
    else:
        return mean([_score_calc(y[i],y_[i]) for i in range(len(y))])

# vector and matrix supportted
def official_score(y,y_):
    assert(dim(y)<=2 and dim(y_)<=2)
    def _score_calc(y,y_):
        numerator = sqrt(mean(square(minus(y,y_))))
        denominator = sqrt(mean(square(y))) + sqrt(mean(square(y_)))
        if denominator==0:
            return 0
        else:
            return 1-(numerator/float(denominator))
    if dim(y) == 1:
        return _score_calc(y,y_)
    else:
        return mean([_score_calc(y[i],y_[i]) for i in range(len(y))])

