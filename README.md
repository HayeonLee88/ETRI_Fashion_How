# 2024 ETRI 자율성장 인공지능 경진대회 Fashion-How
<img width="800" alt="스크린샷 2024-08-25 오후 2 23 13" src="https://github.com/user-attachments/assets/00863043-93a4-414b-8935-9a232f4026df">

<br>

## 제한 사항
- 모델 구성 및 학습 과정은 아래와 같은 제한사항을 준수해야 하며, 준수하지 않을 시 관련 결과는 무효 처리
<img width="800" src="https://github.com/user-attachments/assets/acf0e729-1b79-4c34-83c0-d27f9c9a1db9">

<br>

## 평가기준
- 평가를 위해 리더보드에 제출하는 모델은 '최대 모델 용량' 및 '최소(평가셋) 성능'을 모두 만족해야 하며, 만족하지 않는 경우 관련 결과 무효 처리
- Sub-task 1, 2: '최대 모델 용량' 용량 만족 시, 모델 용량은 순위에 영향을 주지 않으며, 평가 성능이 높은 순으로 순위 산정
- 제출 간 간격은 최소 1시간 이상
<img width="500" src="https://fashion-how.org/images/581d26504510c3691dbdcd515d95c71b-criteria.png">

- 제출 결과는 CPU 환경을 이용하여 추론되며, 사양에 따른 소요 시간은 다음과 같습니다.
<img width="500" src="https://fashion-how.org/images/5660a4371d4727b54dd83fd4b36e9be2-fashionhow4_chart.png">

<br>

## FASCODE (FAShion COordination DatasEt / Fashion CODE)
- Sub-Task 1,2에 제공되는 FASCODE는 **14,000 장의 패션 이미지 데이터**로 구성되어 있으며, 각각의 **의류 위치 정보**를 나타내는 **bounding box 좌표**와 각각의 아이템을 **세 분류의 감성 특징**으로 태깅한 **라벨 정보**가 포함되어 있습니다.
- Sub-Task 3,4에 제공되는 FASCODE는 **옷을 추천해주는 AI 패션 코디네이터와 사용자가 대화를 나눈 데이터셋**으로, AI패션 코디네이터와 사용자가 나눈 대화에 대하여 **발화자 정보, 대화순서, 대화의 기능에 대한 태깅 정보**와 **추천받은 옷에 대한 텍스트 정보**가 담겨있습니다.

<br>

## 👒 Sub-Task 1
- Sub-task 1의 목적은 패션 이미지의 감성 속성들을 추출하고 이에 대 한 정확도를 평가함에 있다.
- 사용하게 될 감성 속성은 일상성, 성, 장식 성 총 3가지로 각각의 속성들은 각기 다른 라벨들을 포함하고 있다.
  - 일상성은 실내복 스타일, 가벼운 외출 스타일, 격식차린 스타일 등 해당 패션이 어떤 상황에 주로 입는지에 따라 분류된 속성이다.
- 해당 태스크에서는 3개의 감성특징들에 대해 각각 속성 분류를 진행하고, 그에 따른 성능을 평가한다.
  - 성능 평가 metric은 top-1 accuracy(%)
  - 전체 성능은 3개의 개별적인 성능을 평균낸 값으로 한다.


<br>

## 👚 Sub-Task 2 | [View Sub-task 2 code](https://github.com/HayeonLee88/ETRI_Fashion_How/tree/001d1bda69859ba2eae2bd0e9f88ad2ec43f9be3/Sub-task2)
- Sub-task 2의 목적은 패션 이미지 분류에서 가장 큰 문제점 중 하나인 불균형 데이터 분류 문제를 해결하기 위함이다.
- 패션 데이터의 특징 카테고리 중 불균형 정도가 심한 카테고리 중 하나인 컬러 특징을 사용한다.
- 성능 평가 metric은 top- 1 accuracy(%)

<br>

## 👔 Sub-Task 3
- Sub-task 3의 목적은 연속 학습을 통해 대화를 토대로 가장 적절한 패션 코디의 순위를 매기는 것이다.
- 성능 평가 metric은 WKT(Weighted Kendall's tau)
  - 켄달타우(Kendall's tau)는 순위 상관 계수(rank correlation coefficient)의 한 종류이며 두 변수들간의 순위를 비교하여 연관성을 계산한다.

※ Sub-task 3에서 WKT로 측정한 성능은 제한사항에서 제시된 '최소(평가셋) 성능'을 만 족하는 지 판단하는데 사용되며, 리더보드 순위는 모델 용량이 작은 순으로 부여된다.

<br>

## 👜 Sub-Task 4
- Sub-task 4의 목적은 제로샷학습을 통해 대화를 토대로 가장 적절한 패션 코디의 순위를 매기는 것이다.
- 성능 평가 metric은 WKT(Weighted Kendall's tau)
  - 켄달타우(Kendall's tau)는 순위 상관 계수(rank correlation coefficient)의 한 종류이며 두 변수들간의 순위를 비교하여 연관성을 계산한다.

※ Sub-task 4에서 WKT로 측정한 성능은 제한사항에서 제시된 '최소(평가셋) 성능'을 만 족하는 지 판단하는데 사용되며, 리더보드 순위는 모델 용량이 작은 순으로 부여된다.
