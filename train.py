import os
import json
import pandas as pd

from tqdm.notebook import tqdm
from gensim.models import Word2Vec
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors
#dataset : train + val + test 의 pd.DataFrame
def run(dataset,min_count=3,size=300,sg=5):

    min_count = min_count
    size = size
    window = max(list(map(len,dataset)))
    sg = 5
    #item2vec 모델 생성
    w2v_model = Word2Vec(total, min_count = min_count, size = size, window = window, sg = sg,seed=1025)
    w2v_model.wv.save('final_wore2vec_for_tag.wv')

    p2v_model = WordEmbeddingsKeyedVectors(size)

if __name__ == '__main__':
    run()
