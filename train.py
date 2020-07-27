import os
import json
import pandas as pd

from tqdm.notebook import tqdm
from gensim.models import Word2Vec
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors
#dataset : train + val + test 의 pd.DataFrame
def item2vec(dataset,min_count=3,size=300,sg=5):
    window = max(list(map(len,dataset)))
    p2v_model = WordEmbeddingsKeyedVectors(size)
    print("start training item2Vec")
    w2v_model = Word2Vec(dataset, min_count = min_count, size = size, window = window, sg = sg,seed=1025)
    w2v_model.save('item2vec.model')
    return w2v_model

if __name__ == '__main__':
    run()
