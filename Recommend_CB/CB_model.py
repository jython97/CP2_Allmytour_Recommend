# -*- coding: utf-8 -*-

## 컨텐츠 기반 추천 모델 ##

## 라이브러리 import

import pandas as pd

import pickle

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity

"""# 데이터 업로드"""

## product 데이터
df = pd.read_pickle('allmytour_proudct.pickle')
df = df.reset_index()

"""## 위도, 경도 유클리드 유사도"""

euclidean_path = 'euclidean_similarity.pickle'

with open(euclidean_path, 'rb') as f:
    euclidean_similarity = pickle.load(f)

"""## 태그 코사인 유사도"""

cosine_path = 'cosine_similarity.pickle'

with open(cosine_path, 'rb') as f:
    cosine_similarity = pickle.load(f)

"""# 거리 + 편의시설 기반 추천 시스템"""

## 인덱스 테이블 만들기
indices = pd.Series(df.index, index=df.product_idx).drop_duplicates()

## 추천 개수 지정
recommandation_num = 10

"""## 거리 기반 추천 모델"""

def recommandation_loc(product_idx, similarity=euclidean_similarity):

    idx = indices[product_idx]
    
    sim_scores = list(enumerate(similarity[idx]))

    sim_scores = sorted(sim_scores, key=lambda x:x[1], reverse = False)

    hotels_idx = [i[0] for i in sim_scores]

    result_df = df.iloc[hotels_idx].copy()
    result_df['loc_score'] = [i[1] for i in sim_scores]

    address2_idx = df.iloc[idx]['address_2_idx']

    if len(result_df.loc[result_df['address_2_idx']==address2_idx]) >= 10:
        result_df = result_df.loc[result_df['address_2_idx']==address2_idx]
    
    result_df = result_df[['product_idx','name_kr','address','loc_score']][1:] 

    return result_df



"""## 태그 기반 추천 모델"""

def recommandation_service(product_idx, similarity=cosine_similarity):

    idx = indices[product_idx] # 입력한 product_idx로 부터 인덱스 가져오기
    address_idx = df.iloc[idx]['address_1_idx']

    sim_scores = list(enumerate(similarity[idx])) # 모든 호텔에 대해서 해당 호텔과의 거리 유사도를 구하기

    sim_scores = sorted(sim_scores, key=lambda x:x[1], reverse = True) # 유사도에 따라 영화들을 정렬

    hotels_idx = [i[0] for i in sim_scores]

    result_df = df.iloc[hotels_idx].copy() # 기존에 읽어들인 데이터에서 해당 인덱스의 값들을 가져온다.
    result_df['tags_score'] = [i[1] for i in sim_scores] # 그리고 스코어 열을 추가하여 코사인 유사도도 확인할 수 있게 한다.

    result_df = result_df.loc[result_df['address_1_idx']==address_idx]
    
    result_df = result_df[['product_idx','star','tags','tags_score','address_2_idx']][1:] # 읽어드린 데이터에서 movieID, 제목, 스코어만 보이게 함

    return result_df



"""## 앙상블"""

def recommandation_loc_service(product_idx):

    loc_df = recommandation_loc(product_idx)

    tags_df = recommandation_service(product_idx)

    result_df = pd.merge(loc_df, tags_df, on='product_idx', how='inner')
    
    result_df['total_score'] =  result_df['tags_score'] - (result_df['loc_score'] * 10)

    result_df = result_df.drop_duplicates()

    result_df = result_df.sort_values(by='total_score', ascending=False)
    
    result_df = result_df[['product_idx','name_kr','star','address','total_score','tags']][1:recommandation_num] # 읽어드린 데이터에서 movieID, 제목, 스코어만 보이게 함

    return result_df

# test
test_product_idx = 6148
recommandation_loc_service(test_product_idx)