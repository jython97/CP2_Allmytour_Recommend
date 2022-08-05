import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.lmf import LogisticMatrixFactorization

#'user_idx', 'product_idx', 'rating' 3 칼럼만 남기도록 data 전처리
def make_data(order_product):
    #중복값 테이블
    dup = order_product.groupby(['user_idx', 'product_idx'], as_index=False).count()[['user_idx', 'product_idx', 'idx']]

    #10회 이상 중복값 user, product 리스트형태로 저장
    dup_user = dup[dup['idx']>=10]['user_idx'].tolist()
    dup_product = dup[dup['idx']>=10]['product_idx'].tolist()

    #user, product, rating 칼럼만 가지도록 테이블을 추출
    data = order_product[['user_idx', 'product_idx', 'order_product_status']].reset_index(drop=True)
    data.rename(columns={'order_product_status':'rating'}, inplace=True)
    data.drop_duplicates(subset=['user_idx', 'product_idx'], inplace=True, keep='last')
    data.reset_index(drop=True, inplace=True)

    #rating칼럼값들을 적절한 정수값으로 바꿔주는 함수
    def make_weight(data, dup_user, dup_product):
        #rating 값들을 1, 2로 만들어주는 함수
        def torate(arg):
            if (arg=='confirm') | (arg=='pending') | (arg=='addpay') | (arg=='complete') & (arg=='fail'):
                arg = 2
            else:
                arg = 1
            return arg

        #10회 이상 중복값의 rating을 특정 rating으로 고정하는 함수
        def fix(table, rating):
            for k, v in zip(dup_user, dup_product):
                table.loc[table[(table['user_idx']==k) & (table['product_idx']==v)].index[0], 'rating'] = rating
            return table

        data['rating'] = data['rating'].apply(torate)
        data = fix(data, 2)
        return data
    
    data = make_weight(data, dup_user, dup_product)
    return data

#user가 구매시도를 했던 모든 상품들을 유저별 리스트형태로 저장
def record_product(data):
    up = data.groupby('user_idx')['product_idx'].apply(list)
    up = up.reset_index()
    return up

#csr_matrix를 만들기 위해 user_idx, product_idx의 인덱스들을 0부터 차례대로 매핑시켜 갱신시킨 값 저장소
def to_idx(data):
    user_to_idx = {v:k for k,v in enumerate(data['user_idx'].unique())}
    product_to_idx = {v:k for k,v in enumerate(data['product_idx'].unique())}
    idx_to_product = {v:k for k,v in product_to_idx.items()}
    idx_to_user = {v:k for k,v in user_to_idx.items()}
    return user_to_idx, product_to_idx, idx_to_product, idx_to_user

#implicit 라이브러리 학습을 위해 필요한 csr_matrix를 만들기 위한 함수
def make_csr(data, user_to_idx, product_to_idx):
    #data 데이터프레임의 'user_idx', 'product_idx'칼럼들의 값을 갱신시키기 위한 함수, 즉 위에서 만든 to_idx를 실행
    def touse(arg):
        return user_to_idx[arg]
    def toproduct(arg):
        return product_to_idx[arg]
    data['user_idx'] = data['user_idx'].apply(touse)
    data['product_idx'] = data['product_idx'].apply(toproduct)

    #csr_matrix를 만들기 위한 행, 열 값 저장
    n_user = data['user_idx'].nunique()
    n_product = data['product_idx'].nunique()
    csr_data = csr_matrix((data.rating, (data.user_idx, data.product_idx)), shape=(n_user, n_product))
    return csr_data

#유저id, 원하는 순위, model종류를 입력하고 csr_data, data, product를 추가적으로 입력해주면 유저에게 추천해주는 데이터프레임을 반환
def recommend_user(user_id, top_n, model, csr_data, user_to_idx, idx_to_product, product):
    user = user_to_idx[user_id]
    recommended = model.recommend(user, csr_data[user], N=top_n, filter_already_liked_items=True)
    return product[product['idx'].isin([idx_to_product[i] for i in recommended[0]])][['idx', 'name_kr', 'avail', 'star', 'address']]

#해당 유저 id 고객이 실제 구매시도한 상품들의 기록을 데이터프레임 형태로 반환
def real_user(user_id, product, user_product):
    return product[product['idx'].isin(user_product[user_product['user_idx']==user_id]['product_idx'].values[0])][['idx', 'name_kr', 'avail', 'star', 'address']]