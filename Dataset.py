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
        sp_model_path = "sp_train"
        print("Start reading json files.....")
        train = pd.read_json('./dataset/train.json', typ = 'frame',encoding='utf-8')
        val = pd.read_json('./dataset/val.json', typ = 'frame',encoding='utf-8')
        test = pd.read_json('./dataset/test.json', typ = 'frame',encoding='utf-8')
        song = pd.read_json('./dataset/song_meta.json', typ='frame',encoding='utf-8')
        print("done.\n")
        #######################################################################

        plylst_tag = train['tags']
        tag_counter = Counter([tg for tgs in plylst_tag for tg in tgs])
        tag_dict = {x: tag_counter[x] for x in tag_counter}

        tag_id_tid = dict()
        tag_tid_id = dict()
        for i, t in enumerate(tag_dict):
            tag_id_tid[t] = i
            tag_tid_id[i] = t
        n_tags = len(tag_dict)
        self.tag_tid_id = tag_tid_id

        plylst_song = train['songs']
        song_dict = {x: x for x in song['id']}

        n_songs = len(song_dict)

        train['tags_id'] = train['tags'].map(lambda x: [tag_id_tid.get(t) for t in x if tag_id_tid.get(t) != None])

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
            if i%10000 == 0:
                print(i)
            counter = Counter(train.loc[index]['plylst_genre_id'])
            for (k,c) in counter.items():
                gnr_array[i][k] = c
        print("done. \n")
        #######################################################################

        plylst_use = train[['plylst_title','updt_date','tags_id','songs']]
        plylst_use.loc[:,'num_songs'] = plylst_use['songs'].map(len)
        plylst_use.loc[:,'num_tags'] = plylst_use['tags_id'].map(len)
        #######################################################################
        print("start applying sentencepiece to plylst_title....")
        plylst_train = plylst_use
        if "{}.model".format(sp_model_path) not in os.listdir():
            makeSentencepieceModel(train,sp_model_path)
        sp = SentencePieceProcessor()
        sp.Load("{}.model".format(sp_model_path))

        cv = CountVectorizer(max_features=3000, tokenizer=sp.encode_as_pieces)
        content = train['plylst_title']
        tdm = cv.fit_transform(content)

        title_tdm = tdm.toarray()
        print("done. \n")
        #######################################################################

        n_train = len(plylst_train)
        row = np.repeat(range(n_train), plylst_train['num_songs']) # User Index 별 노래 개수만큼 만듦
        col = [song for songs in plylst_train['songs'] for song in songs] # Song dic number 추출
        dat = np.repeat(1, plylst_train['num_songs'].sum()) # User별 Song이 있는 부분에 1을 넣기위해 1과 전체 노래 개수만큼 만듦
        train_user_songs_A = spr.csr_matrix((dat, (row, col)), shape=(n_train, n_songs)) # csr_matrix 제작

        row = np.repeat(range(n_train), plylst_train['num_tags'])
        col = [tag for tags in plylst_train['tags_id'] for tag in tags]
        dat = np.repeat(1, plylst_train['num_tags'].sum())
        train_user_tags_A = spr.csr_matrix((dat, (row, col)), shape=(n_train, n_tags))

        train_user_songs_A_T = train_user_songs_A.T.tocsr()
        train_user_songs_A_T # 행에는 노래 columns에는 User 정보 삽입

        train_user_tags_A_T = train_user_tags_A.T.tocsr()
        train_user_tags_A_T # 행에는 Tangs columns에는 User 정보 삽입

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
            if i%10000 == 0:
                print(i)
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
        self.spr_list = [train_user_songs_A,train_user_tags_A,train_user_songs_A_T,train_user_tags_A_T,\
                test_title,title_sp,gnr_sp,test_gnr_sp,\
                title_gnr,test_title_gnr]
