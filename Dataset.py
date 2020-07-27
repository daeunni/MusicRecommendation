import os,time
import pandas as pd
import numpy as np
from collections import Counter
from sentencepiece import SentencePieceTrainer
from sentencepiece import SentencePieceProcessor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import vstack
from scipy import sparse
import scipy.sparse as spr
from util import write_json,makeSentencepieceModel
from sklearn.feature_extraction.text import CountVectorizer

class Dataset:
    def __init__(self):
        #######################################################################
        sp_train_model_path = "sp_train"
        sp_total_model_path = "sp_total"
        print("Start reading json files.....")
        train = pd.read_json('./dataset/train.json', typ = 'frame',encoding='utf-8')
        val = pd.read_json('./dataset/val.json', typ = 'frame',encoding='utf-8')
        test = pd.read_json('./dataset/test.json', typ = 'frame',encoding='utf-8')
        song = pd.read_json('./dataset/song_meta.json', typ='frame',encoding='utf-8')
        total = pd.concat([train,val,test])
        total.index = range(len(total))
        self.total = total
        print("done.\n")

        song_dict = {x: x for x in song['id']}
        n_songs = len(song_dict)

        tag_counter = Counter([tg for tgs in train['tags'] for tg in tgs])
        tag_dict = {x: tag_counter[x] for x in tag_counter}

        tag_id_tid = dict()
        tag_tid_id = dict()
        for i, t in enumerate(tag_dict):
            tag_id_tid[t] = i
            tag_tid_id[i] = t
        n_tags = len(tag_dict)
        self.tag_tid_id = tag_tid_id
        train['tags_id'] = train['tags'].map(lambda x: [tag_id_tid.get(t) for t in x if tag_id_tid.get(t) != None])
        #######################################################################

        #######################################################################
        # song genre 내용 가져오기.
        print("start making genre array....")
        song_cate = []

        for i in range(len(train)):
            gnr = []
            songs = train.iloc[i,3]

            for j in songs:
                for k in song.iloc[j,0]:
                    gnr.append(k)
            song_cate.append(gnr)

        train['plylst_genre'] = song_cate

        plylst_genre = train['plylst_genre']
        genre_counter = Counter([gen for genre in plylst_genre for gen in genre])
        genre_dict = {x: genre_counter[x] for x in genre_counter}

        genre_id_tid = dict()
        genre_tid_id = dict()
        for i, t in enumerate(genre_dict):
            genre_id_tid[t] = i
            genre_tid_id[i] = t
        n_genre = len(genre_dict)
        train['plylst_genre_id'] = train['plylst_genre'].map(lambda x: [genre_id_tid.get(s) for s in x if genre_id_tid.get(s) != None])

        gnr_array = np.zeros((len(train),n_genre))
        for i,index in enumerate(train.index):
            counter = Counter(train.loc[index]['plylst_genre_id'])
            for (k,c) in counter.items():
                gnr_array[i][k] = c
        print("done. \n")
        #######################################################################
        train.loc[:,'num_songs'] = train['songs'].map(len)
        train.loc[:,'num_tags'] = train['tags_id'].map(len)
        #######################################################################
        print("start applying sentencepiece to plylst_title....")

        cv, title_tdm = make_title_tdm(train,sp_train_model_path)
        #total_title_tdm = make_title_tdm(total,sp_total_model_path)

        print("done. \n")
        #######################################################################
        train_user_songs_A = make_sparse(train,train['songs'],n_songs)
        train_user_tags_A = make_sparse(train,train['tags_id'],n_tags)

        #######################################################################
        song_cate = []
        for i in range(len(test)):
            gnr = []
            songs = test.iloc[i,3]

            for j in songs:
                for k in song.iloc[j,0]:
                    gnr.append(k)
            song_cate.append(gnr)

        test['plylst_genre'] = song_cate

        test['tags_id'] = test['tags'].map(lambda x: [tag_id_tid.get(t) for t in x if tag_id_tid.get(t) != None])
        test['plylst_genre_id'] = test['plylst_genre'].map(lambda x: [genre_id_tid.get(s) for s in x if genre_id_tid.get(s) != None])
        test_title = cv.transform(test['plylst_title']).toarray()

        gnr_val = np.zeros((len(test),n_genre))
        for i,index in enumerate(test.index):
            counter = Counter(test.loc[index]['plylst_genre_id'])
            for (k,c) in counter.items():
                gnr_val[i][k] = c
        test_title_gnr = np.concatenate((gnr_val,test_title),axis=1)
        #######################################################################

        title_gnr = np.concatenate((gnr_array,title_tdm),axis=1)
        title_gnr = sparse.csr_matrix(title_gnr)

        title_sp = sparse.csr_matrix(title_tdm)

        gnr_sp = sparse.csr_matrix(gnr_array)
        test_gnr_sp = sparse.csr_matrix(gnr_val)

        self.test = test
        self.n_tags = n_tags
        self.n_songs = n_songs
        self.spr_list = [train_user_songs_A,train_user_tags_A,\
                test_title,title_sp,gnr_sp,test_gnr_sp,\
                title_gnr,test_title_gnr]

def make_title_tdm(df,path):
    if "{}.model".format(path) not in os.listdir():
        makeSentencepieceModel(df,path)
    sp = SentencePieceProcessor()
    sp.Load("{}.model".format(path))

    cv = CountVectorizer(max_features=3000, tokenizer=sp.encode_as_pieces)
    content = df['plylst_title']
    tdm = cv.fit_transform(content)

    title_tdm = tdm.toarray()
    return cv,title_tdm

def make_sparse(df,df_item,n_item):
    n_df = len(df)
    row = np.repeat(range(n_df), df_item.apply(len)) # User Index 별 노래 개수만큼 만듦
    col = [item for items in df_item for item in items] # Song dic number 추출
    dat = np.repeat(1, df_item.apply(len).sum()) # User별 Song이 있는 부분에 1을 넣기위해 1과 전체 노래 개수만큼 만듦
    spr_matrix = spr.csr_matrix((dat, (row, col)), shape=(n_df, n_item)) # csr_matrix 제작
    return spr_matrix
