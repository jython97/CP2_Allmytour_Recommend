import work
import model
import pickle
import pandas as pd
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.lmf import LogisticMatrixFactorization

product = pd.read_excel('allmytour_DB_data.xlsx', sheet_name='product')
order_product = pd.read_excel('allmytour_DB_data.xlsx', sheet_name='order_product')

data = work.make_data(order_product)
user_product = work.record_product(data)
user_to_idx, product_to_idx, idx_to_product, idx_to_user = work.to_idx(data)
csr_data = work.make_csr(data, user_to_idx, product_to_idx)

#모델 학습 예시
# als = model.ALS(csr_data)
# bpr = model.BPR(csr_data)
# lmf = model.LMF(csr_data)

#lmf의 경우 iterations=100000 학습시킨 모델은 시간이 1시간 반 이상이 걸리므로 피클링 파일로 미리 저장해놓음
#lmf:100000번, lmf2:500000번
with open('lmf.pickle', 'rb') as l:
    lmf = pickle.load(l)

#유저 id = 10, top 5 lmf 추천 리스트
print(work.recommend_user(10, 5, lmf, csr_data, user_to_idx, idx_to_product, product))
print("----------------------------------------------------------------------------------------------")
#유저 id = 10 실제 구매시도 리스트
print(work.real_user(10, product, user_product))