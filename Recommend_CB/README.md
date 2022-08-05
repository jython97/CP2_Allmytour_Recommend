# Recommend_CB

상품 tag, 상품 간 거리와 Countvectorize & Cosine-Similarity 를 활용한 컨텐츠 기반 추천 파이썬 파일입니다.  

- **allmytour_proudct.pickle** : 제공된 DB와 크롤링한 tag가 포함된 피클 파일
- **euclidean_similarity.pickle** : 호텔 간 유클리드 거리 유사도가 저장된 피클 파일  
- **cosine_similarity.pickle** : 호텔 간 코사인 유사도가 저장된 피클 파일
- **CB_model.py** : 거리 기반, 태그 기반, 거리 + 태그 기반 모델이 저장된 파이썬 파일
