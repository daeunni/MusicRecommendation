# Makgora team 

1. requirements

* sentencepiece
* scipy
* gensim
* tqdm
* sklearn
* numpy
* pandas

## Usage

```

```





## Song 추천

1. Song meta에 있는 모든 정보들을 활용하여 train과 valid, test간 유사도를 반영하였다. 
2. Knn을 통해 playlist간 유사도가 높은 상위 25개의 플레이리스트를 선정하여 CF를 진행하였다. 
3. 이 경우, 10개 내외로 playlist가 채워지지 않는 현상이 있었는데, 이는 k=50일 경우를 통해 채웠다. 이를 활용하여 song 정보를 삽입 하였다.

## Tag 추천

1. Test 데이터에 song 정보와 tag 정보가 모두 있을 경우 이 두개의 정보와 플레이리스트 이름, 노래 장르정보를 활용하여 CF를 진행. 
2. Song이 없는 경우 tags 정보와 title 정보를 활용해서 CF를 진행하였고 tag정보가 없는 경우는 song과 장르, title 정보를 활용하여 song과 tag를 추천하는 모델을 작성하였다. 둘다 없는 경우는 Title 정보를 활용
3. 제목 정보에서 3번 이상한 태그 정보를 추출해 태그에 추가
4. 한 플레이리스트에 노래 + 태그를 넣어 Item2Vec 진행
5. Song / Tag 정보가 없거나 부족한 경우 2번에서 구한 Tag값을 가져옴