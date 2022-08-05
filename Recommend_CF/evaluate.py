#원본 데이터에서 긍정적인 피드백을 정해진 비율만큼 부정적 피드백으로 만든 것을 trainset으로 만들고 원본을 testset으로 만듬.
import random
import numpy as np
import pandas as pd
from sklearn import metrics
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.lmf import LogisticMatrixFactorization
from model import ALS
import work

def make_train (matrix, percentage = .2):
    '''
    -----------------------------------------------------
    설명
    유저-아이템 행렬 (matrix)에서 
    1. 0 이상의 값을 가지면 1의 값을 갖도록 binary하게 테스트 데이터를 만들고
    2. 훈련 데이터는 원본 행렬에서 percentage 비율만큼 0으로 바뀜
    
    -----------------------------------------------------
    반환
    training_set: 훈련 데이터에서 percentage 비율만큼 0으로 바뀐 행렬
    test_set:     원본 유저-아이템 행렬의 복사본
    user_inds:    훈련 데이터에서 0으로 바뀐 유저의 index
    '''
    test_set = matrix.copy()
    test_set[test_set !=0] = 1 # binary하게 만들기
    
    training_set = matrix.copy()
    nonzero_inds = training_set.nonzero()
    nonzero_pairs = list(zip(nonzero_inds[0], nonzero_inds[1]))
    
    random.seed(0)
    num_samples = int(np.ceil(percentage * len(nonzero_pairs)))
    samples = random.sample (nonzero_pairs, num_samples)
    
    user_inds = [index[0] for index in samples]
    item_inds = [index[1] for index in samples]
    
    training_set[user_inds, item_inds] = 0
    training_set.eliminate_zeros()
    
    return training_set, test_set, list(set(user_inds))

#성능 평가는 AUC
def auc_score (test, predictions):
    '''
    fpr, tpr를 이용해서 AUC를 계산하는 함수
    '''
    fpr, tpr, thresholds = metrics.roc_curve(test, predictions)
    return metrics.auc(fpr,tpr)

def calc_mean_auc(training_set, altered_users, predictions, test_set):
    '''
    가려진 정보가 있는 유저마다 AUC 평균을 구하는 함수
    ----------------------------------------
    input
    1. training_set: make_train 함수에서 만들어진 훈련 데이터 (일정 비율로 아이템 구매량이 0으로 가려진 데이터)
    2. prediction: implicit MF에서 나온 유저/아이템 별로 나온 예측 평점 행렬
    3. altered_users: make_train 함수에서 아이템 구매량이 0으로 가려진 유저
    4. test_set: make_train함수에서 만든 테스트 데이터
    ----------------------------------------
    반환
    추천 시스템 유저의 평균 auc
    인기아이템 기반 유저 평균 auc
    '''
    # 리스트 초기화
    store_auc = []
    popularity_auc = []
    
    pop_items = np.array(test_set.sum(axis = 0)).reshape(-1) # 모든 유저의 아이템별 구매횟수 합
    item_vecs = predictions[1] # 아이템 latent 벡터
    
    for user in altered_users:
        training_row = training_set[user,:].toarray().reshape(-1) # 유저의 훈련데이터
        zero_inds = np.where(training_row == 0) # 가려진 아이템 Index
        
        # 가려진 아이템에 대한 예측
        user_vec = predictions[0][user,:]
        pred = user_vec.dot(item_vecs).toarray()[0,zero_inds].reshape(-1)
        
        # 가려진 아이템에 대한 실제값
        actual = test_set[user,:].toarray()[0,zero_inds].reshape(-1) 
        
        # 가려진 아이템에 대한 popularity (구매횟수 합)
        pop = pop_items[zero_inds]
        
        # AUC 계산 
        store_auc.append(auc_score(actual, pred))
        popularity_auc.append(auc_score(actual,pop))
    
    return float('%.3f'%np.mean(store_auc)), float('%.3f'%np.mean(popularity_auc))

#성능평가 예시(ALS)
# product = pd.read_excel('allmytour_DB_data.xlsx', sheet_name='product')
# order_product = pd.read_excel('allmytour_DB_data.xlsx', sheet_name='order_product')
# data = work.make_data(order_product)
# user_product = work.record_product(data)
# user_to_idx, product_to_idx, idx_to_product, idx_to_user = work.to_idx(data)
# csr_data = work.make_csr(data, user_to_idx, product_to_idx)
# train, test, product_users_altered = make_train(csr_data, 0.2)
# als = ALS(train)
# predictions = [csr_matrix(als.user_factors), csr_matrix(als.item_factors.T)]
# print(calc_mean_auc(train,product_users_altered, predictions, test))