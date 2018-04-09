from linalg.common import sqrt,mean,square,minus,mean,dim

# vector and matrix supportted
def l2_loss(y,y_,return_losses=False):
    assert(dim(y)<=2 and dim(y_)<=2)
    def _score_calc(y,y_):
        numerator = sqrt(mean(square(minus(y,y_))))
        return numerator

    if dim(y) == 1:
        return _score_calc(y,y_)
    else:
        losses = [_score_calc(y[i],y_[i]) for i in range(len(y))]
        if return_losses:
            return losses
        else:
            return mean(losses)

# vector and matrix supportted
def official_score(y,y_,return_scores=False):
    y_ = [int(round(i)) for i in y_]
    
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
        scores = [_score_calc(y[i],y_[i]) for i in range(len(y))]
        if return_scores:

            return scores
        else: 
            return mean(scores)

