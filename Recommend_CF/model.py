from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.lmf import LogisticMatrixFactorization

def ALS(data):
    als = AlternatingLeastSquares(factors=50, regularization=100, use_gpu=False, iterations=20)
    als.fit(data)
    return als

def BPR(data):
    bpr = BayesianPersonalizedRanking(factors=100, learning_rate=0.01, regularization=0.01, iterations=1, verify_negative_samples=True, random_state=42)
    bpr.fit(data)
    return bpr

def LMF(data):
    lmf = LogisticMatrixFactorization(factors=50, learning_rate=0.01, regularization=0.01, iterations=100000, random_state=42)
    lmf.fit(data)
    return lmf