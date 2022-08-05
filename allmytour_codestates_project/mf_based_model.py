""" Excel data만 빠르게 호출할 수 있는 방법 찾은 후 진행할 것을 권장.
    *참고: 작성된 함수들은 진행 순서에 맞게 작성 됨
    
    [함수 설명]
    - 추천 출력 값과 활용 할 함수
    get_img: DATA SET에 저장 된 상품 img 정보 가져오는 함수
    product_info: 추천 시 상품명과 함께 제공되는 세부 정보제공 함수
    
    - MF 알고리즘에 넣을 input 데이터 처리 함수
    try_payment_product: 결제시도 1번 이상된 상품 DATA SET 생성 함수
    payment_confirmed_product: 최종 결제 완료된 상품 DATA SET 생성 함수. try_paryment_product return 값을 인자로 받음
    input_product_preprocessing: 추천 모델에 사용가능하도록 (try_payment_product) or (payment_confirmed_product)의 DATA SET 전처리 함수

    - MF 알고리즘 함수
    compute_cos_similarity: 코사인 유사도 계산하는 함수
    recommend_model: click한 상품 index와 전처리 된 상품 데이터를 통해 click한 상품과 유사한 상품 추천 함수, 
                     (유사도 계산 시 알고리즘에 활용되지 않은 상품일 경우 랜덤으로 상품 추천) """


""" - 추천 시스템 평가지표 함수
    Recall: 실제 구매한 상품의 수 대비 추천해서 구매한 상품의 비율을 의미합니다.
    Precision: 추천한 상품의 수 대비 추천해서 구매한 상품의 비율을 의미합니다.
    F1_score: 위 2가지 함수를 함께 봄으로써 추천 성능과 현실 성과를 적절히 조율하여 판단하는 함수입니다. 
    
    추천 시스템 평가지표 함수는 추후에 추천모델 적용 후 A/B test할 때 진행하면 됩니다. """
    
import pandas as pd
pd.set_option('mode.chained_assignment',  None)
import numpy as np
import random 
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from bs4 import BeautifulSoup
from sklearn.decomposition import randomized_svd
random.seed(0)

### DATA LOAD ###
sheet_names = ['order_product','product_room','product_detail_kr','product']
try_payment_product = pd.read_excel('./allmytour_DB_data.xlsx',  sheet_name=sheet_names)
order_product = try_payment_product[sheet_names[0]]
product_room = try_payment_product[sheet_names[1]]
product_detail_kr = try_payment_product[sheet_names[2]]
product = try_payment_product[sheet_names[3]]
# Excel file loading 시간 단축 필요(==> 서버형DB 연결)

# 필요한 특성만 추출
product = product[['idx','name_kr','type','avail','main_view','star','address','address_1_idx']]
product_detail_kr = product_detail_kr[['product_idx', 'info_detail','info_basic']]
product_room = product_room[['product_idx','roomtype_idx','name']]
order_product = order_product[['product_idx','order_num','order_product_status','user_idx','terms','adt_cnt','chd_cnt','room_cnt']]
# 상품 인덱스명 통일
product.rename({'idx':'product_idx'}, axis=1, inplace=True)

def get_img(x):
    """img만 추출"""
    html = x
    
    try:
        if html == None:
            return '-'
        elif html == '-':
            return html
        else:
            soup = BeautifulSoup(html)
            x = soup.find('img')['src']
            return x
    except:
        return '-'

def product_info(sheet1=product, sheet2=product_detail_kr, sheet3=product_room, shee4=order_product):
    """추천 리스트 결과와 함께 넘길 DATA SET"""
    
    # 유저 별 상품과 관련된 모든 정보:all_product
    df1 = pd.merge(product, product_detail_kr, on='product_idx',how='left')
    df2 = pd.merge(df1, product_room, on='product_idx', how='left')
    all_product = pd.merge(df2, order_product, on='product_idx', how='left')
    
    # 추천 리스트와 함께 넘길 DATA SET. 여기서 index ==> product_idx
    product_info = all_product[['product_idx','name_kr','address_1_idx','info_img']].drop_duplicates().reset_index(drop=True)
    product_info.set_index('product_idx', inplace=True)
    
    return product_info, all_product

def try_payment_product(all_product):
    """현재 사용 가능한 상품 & 결제시도 1번이라도 된 상품 DATA SET"""
    condition = (all_product['avail'] == 1) 
    try_payment_product = all_product.loc[condition, ['product_idx','name_kr','star', 'address', 'address_1_idx','order_product_status','terms','room_cnt','adt_cnt','chd_cnt']].drop_duplicates(subset='name_kr').reset_index(drop=True)
    
    return try_payment_product

def payment_confirmed_product(try_payment_product):
    """현재 사용 가능한 상품 & 결제 완료 된 상품 DATA SET"""
    """try_payment_product method에서 return 된 try_payment_product SET을 arg*로 활용"""
    payment_confirmed_product = try_payment_product.loc[try_payment_product['order_product_status'] == 'confirm']
    
    return payment_confirmed_product

def input_product_preprocessing(df):
    """address column 삭제, np.nan → 각 column의 최빈값으로 대체, star만 3.0으로 대체"""
    df.drop(['address'], axis=1, inplace=True)
    
    # NaN 값 처리. 추후 변경 가능
    df['adt_cnt'] = df['adt_cnt'].fillna(2.0)
    df['chd_cnt'] = df['chd_cnt'].fillna(0.0)
    df['room_cnt'] = df['room_cnt'].fillna(1.0)
    df['terms'] = df['terms'].fillna(1.0)
    df['star'] = df['star'].fillna(3.0)
    
    df = pd.get_dummies(df, drop_first=True, columns=['address_1_idx', 'order_product_status'])
    
    df = df.set_index('product_idx').drop('name_kr', axis=1)
    
    return df

def compute_cos_similarity(v1, v2):
    norm1 = np.sqrt(np.sum(np.power(v1, 2)))
    norm2 = np.sqrt(np.sum(np.power(v2, 2)))
    dot = np.dot(v1, v2)
    
    return dot / (norm1 * norm2)

def recommend_model(df, click_idx):
    """MF model algorithm, (만약 최종 결제 완료 된 상품 기반 추천을 원할 시 모델 수정 필요)"""
    if click_idx in df.index.values:
        U_5, S_5, V_5 = randomized_svd(df.values, n_components=5, random_state=0) 
        item_idx = df.index
        temp = pd.DataFrame(U_5, index=item_idx)
        click_vector = temp.loc[click_idx]   
     
        # 호텔 & 유사도 리스트
        hotels_with_scores = []
        for hotel_idx, hotel_vector in zip(temp.index, temp.values):
            if click_idx != hotel_idx:
                cos_similarity = compute_cos_similarity(click_vector, hotel_vector)
                hotels_with_scores.append((hotel_idx, cos_similarity))
    
        # 상품 유사도 상위 4% 목록
        hotel_score = pd.DataFrame(hotels_with_scores, columns=['hotel_idx', 'cos_similarity'])
        condition = (hotel_score['cos_similarity'] >= hotel_score['cos_similarity'].quantile(.96))
        top_hotels = hotel_score[condition]
    
        recommend_5 = top_hotels.sort_values(by='cos_similarity', ascending=False).head(5)['hotel_idx']
        return recommend_5.values # 추천상품 index를 return 합니다.
    
    else: # 유사도 계산 시 알고리즘에 활용되지 않은 상품일 경우 랜덤으로 상품 추천
        click_idx = np.random.choice(df.index.values, 1, replace=False)[0]
        U_5, S_5, V_5 = randomized_svd(df.values, n_components=5, random_state=0)
    
        item_idx = df.index
        temp = pd.DataFrame(U_5, index=item_idx)
        click_vector = temp.loc[click_idx]   
     
        # 호텔 & 유사도 리스트
        hotels_with_scores = []
        for hotel_idx, hotel_vector in zip(temp.index, temp.values):
            if click_idx != hotel_idx:
                cos_similarity = compute_cos_similarity(click_vector, hotel_vector)
                hotels_with_scores.append((hotel_idx, cos_similarity))
    
        # 상품 유사도 상위 4% 목록
        hotel_score = pd.DataFrame(hotels_with_scores, columns=['hotel_idx', 'cos_similarity'])
        condition = (hotel_score['cos_similarity'] >= hotel_score['cos_similarity'].quantile(.96))
        top_hotels = hotel_score[condition]
    
        recommend_5 = top_hotels.sort_values(by='cos_similarity', ascending=False).head(5)['hotel_idx']
        return recommend_5.values # 추천상품 index를 return 합니다.


def precision(total_recommend_list, recommend_and_purchased_list):
    """추천한 상품 중에 실제 구매로 이어진 경우 확인"""
    recommend_count = len(total_recommend_list)
    recommend_and_purchased_count = len(recommend_and_purchased_list)
    
    return recommend_and_purchased_count / recommend_count
    
def recall(total_purchased_list, recommend_and_purchased_list):
    """구매된 상품 중에 추천으로 구매된 상품 비율 확인"""
    purchased_count = len(total_purchased_list)
    recommend_and_purchased_count = len(recommend_and_purchased_list)
    
    return recommend_and_purchased_count / purchased_count

def f1_score(precision, recall):
    """precision와 recall 조화 평균"""
    return 2 / ((1 / precision) + (1 / recall))




### 모델 활용 순서는 아래와 같습니다 ###

# img 링크만 가져오기, np.nan → '-' 변경
product_detail_kr['info_detail'] = product_detail_kr['info_detail'].fillna('-')

# 이미지 링크 추출한 값 info_img 컬럼으로 추가
product_detail_kr['info_img'] = product_detail_kr['info_detail'].apply(get_img)

# idx, name, address_idx, img DATA SET 생성
product_info, all_product = product_info()

# 결제 시도 1번 이상 시도한 DATA SET = try_payment_product
try_payment_product = try_payment_product(all_product)

# model input DATA 생성 완료
try_payment_product = input_product_preprocessing(try_payment_product)

# test_product 설정
test_idx = 49
test_product_info = product_info.loc[test_idx]

# recommendation test ---> try_payment_product DATA SET
print('최종 추천 리스트 idx')
display(recommend_model(try_payment_product, test_idx))
print('-'*100)
print('테스트 아이템 정보')
display(pd.DataFrame(test_product_info).T)
print('-'*100)
print('추천 리스트 상품정보')
recommend_list = recommend_model(try_payment_product, test_idx)
display(product_info.loc[recommend_list])


    
