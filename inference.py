from datetime import timedelta, datetime
import glob
import json
import os
import re
import pickle

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
from scipy.sparse import vstack
from scipy import sparse
from util import write_json,makeSentencepieceModel
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm_notebook
from sklearn.neighbors import NearestNeighbors

from Dataset import Dataset
import pre_tag,word2vec_for_tag

def song_inference():
    sp_total_model_path = "sp_total"
    train = pd.read_json('./dataset/train.json', typ = 'frame',encoding='utf-8')
    song = pd.read_json('./dataset/song_meta.json', typ = 'frame',encoding='utf-8')
    plylst_tag = train['tags']
    tag_counter = Counter([tg for tgs in plylst_tag for tg in tgs])
    tag_dict = {x: tag_counter[x] for x in tag_counter}

    tag_id_tid = dict()
    tag_tid_id = dict()
    for i, t in enumerate(tag_dict):
        tag_id_tid[t] = i
        tag_tid_id[i] = t
    n_tags = len(tag_dict)

    plylst_song = train['songs']
    song_dict = {x: x for x in song['id']}

    n_songs = len(song_dict)

    train['tags_id'] = train['tags'].map(lambda x: [tag_id_tid.get(t) for t in x if tag_id_tid.get(t) != None])
    # song genre 내용 가져오기.
    song_cate = []

    for i in range(len(train)):
        gnr = []
        songs = train.iloc[i,3]

        for j in songs:
            for k in song.loc[j,'song_gn_dtl_gnr_basket']:
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
    gnr_array.shape

    song['issue_date'] = song['issue_date'].astype('str').map(lambda x : x[:6])

    plylst_use = train[['plylst_title','updt_date','tags_id','songs']]
    plylst_use.loc[:,'num_songs'] = plylst_use['songs'].map(len)
    plylst_use.loc[:,'num_tags'] = plylst_use['tags_id'].map(len)

    plylst_train = plylst_use

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

    val = pd.read_json('./dataset/val.json', typ = 'frame',encoding='utf-8')

    song_cate = []

    for i in range(len(val)):
        gnr = []
        songs = val.iloc[i,3]

        for j in songs:
            for k in song.loc[j,'song_gn_dtl_gnr_basket']:
                gnr.append(k)
        song_cate.append(gnr)

    val['plylst_genre'] = song_cate

    val['tags_id'] = val['tags'].map(lambda x: [tag_id_tid.get(t) for t in x if tag_id_tid.get(t) != None])
    val['plylst_genre_id'] = val['plylst_genre'].map(lambda x: [genre_id_tid.get(s) for s in x if genre_id_tid.get(s) != None])
    val.loc[:,'num_songs'] = val['songs'].map(len)
    val.loc[:,'num_tags'] = val['tags_id'].map(len)
    # val_title = cv.transform(val['plylst_title']).toarray()

    gnr_val = np.zeros((len(val),n_genre))
    for i,index in enumerate(val.index):
        if i%10000 == 0:
            print(i)
        counter = Counter(val.loc[index]['plylst_genre_id'])
        for (k,c) in counter.items():
            gnr_val[i][k] = c
    gnr_val.shape

    n_val = len(val)
    row = np.repeat(range(n_val), val['num_songs']) # User Index 별 노래 개수만큼 만듦
    col = [song for songs in val['songs'] for song in songs] # Song dic number 추출
    dat = np.repeat(1, val['num_songs'].sum()) # User별 Song이 있는 부분에 1을 넣기위해 1과 전체 노래 개수만큼 만듦
    val_user_songs_A = spr.csr_matrix((dat, (row, col)), shape=(n_val, n_songs)) # csr_matrix 제작

    row = np.repeat(range(n_val), val['num_tags'])
    col = [tag for tags in val['tags_id'] for tag in tags]
    dat = np.repeat(1, val['num_tags'].sum())
    val_user_tags_A = spr.csr_matrix((dat, (row, col)), shape=(n_val, n_tags))

    val_user_songs_A_T = val_user_songs_A.T.tocsr()
    val_user_tags_A_T = val_user_tags_A.T.tocsr()

    test = pd.read_json('./dataset/test.json', typ = 'frame',encoding='utf-8')

    song_cate = []

    for i in range(len(test)):
        gnr = []
        songs = test.iloc[i,3]

        for j in songs:
            for k in song.loc[j,'song_gn_dtl_gnr_basket']:
                gnr.append(k)
        song_cate.append(gnr)

    test['plylst_genre'] = song_cate

    test['tags_id'] = test['tags'].map(lambda x: [tag_id_tid.get(t) for t in x if tag_id_tid.get(t) != None])
    test['plylst_genre_id'] = test['plylst_genre'].map(lambda x: [genre_id_tid.get(s) for s in x if genre_id_tid.get(s) != None])
    test.loc[:,'num_songs'] = test['songs'].map(len)
    test.loc[:,'num_tags'] = test['tags_id'].map(len)
    # test_title = cv.transform(test['plylst_title']).toarray()

    gnr_test = np.zeros((len(test),n_genre))
    for i,index in enumerate(test.index):
        if i%10000 == 0:
            print(i)
        counter = Counter(test.loc[index]['plylst_genre_id'])
        for (k,c) in counter.items():
            gnr_test[i][k] = c
    gnr_test.shape

    n_test = len(test)
    row = np.repeat(range(n_test), test['num_songs']) # User Index 별 노래 개수만큼 만듦
    col = [song for songs in test['songs'] for song in songs] # Song dic number 추출
    dat = np.repeat(1, test['num_songs'].sum()) # User별 Song이 있는 부분에 1을 넣기위해 1과 전체 노래 개수만큼 만듦
    test_user_songs_A = spr.csr_matrix((dat, (row, col)), shape=(n_test, n_songs)) # csr_matrix 제작

    row = np.repeat(range(n_test), test['num_tags'])
    col = [tag for tags in test['tags_id'] for tag in tags]
    dat = np.repeat(1, test['num_tags'].sum())
    test_user_tags_A = spr.csr_matrix((dat, (row, col)), shape=(n_test, n_tags))

    test_user_songs_A_T = test_user_songs_A.T.tocsr()
    test_user_tags_A_T = test_user_tags_A.T.tocsr()

    data_all = pd.concat([train,val,test])
    data_all.index = range(len(data_all))

    arts = song['artist_id_basket'].map(lambda x : x[0])

    arts = pd.DataFrame(arts)

    art_counts = arts['artist_id_basket'].value_counts().reset_index()
    art_counts.columns = ['artist_id_basket','counts']

    arts2 = pd.merge(arts,art_counts,how='left',on=['artist_id_basket'])

    song_art = song.iloc[arts2.query('counts >= 12')['artist_id_basket'].index]

    song_art = song_art[['artist_id_basket']]

    #아티스트 대분류
    ART_cate = []

    for i in tqdm_notebook(range(len(data_all))):
        ART = []
        songs = data_all.loc[i,'songs']

        for j in songs:
          if j in song_art.index :
            for k in song_art.loc[j,'artist_id_basket'] :
                ART.append(k)
        ART_cate.append(ART)


    data_all['plylst_ARTIST'] = ART_cate

    plylst_ARTIST = data_all['plylst_ARTIST']
    ARTIST_counter = Counter([ART for ARTIST in plylst_ARTIST for ART in ARTIST])
    ARTIST_dict = {x: ARTIST_counter[x] for x in ARTIST_counter}

    ARTIST_id_tid = dict()
    ARTIST_tid_id = dict()
    for i, t in enumerate(ARTIST_dict):
        ARTIST_id_tid[t] = i
        ARTIST_tid_id[i] = t
    n_ARTIST = len(ARTIST_dict)
    data_all['plylst_ARTIST_id'] = data_all['plylst_ARTIST'].map(lambda x: [ARTIST_id_tid.get(s) for s in x if ARTIST_id_tid.get(s) != None])

    ART_data_all = np.zeros((len(data_all),n_ARTIST))
    for i,index in enumerate(data_all.index):
        if i%10000 == 0:
            print(i)
        counter = Counter(data_all.loc[index]['plylst_ARTIST_id'])
        for (k,c) in counter.items():
            ART_data_all[i][k] = c
    ART_data_all.shape

    ART_array = ART_data_all[:len(train)]
    ART_val = ART_data_all[len(train):len(train)+len(val)]
    ART_test = ART_data_all[len(train)+len(val):len(train)+len(val)+len(test)]


    # ART_data_all = sparse.csr_matrix(ART_data_all)
    del ART_data_all

    ART_array = sparse.csr_matrix(ART_array)
    ART_val = sparse.csr_matrix(ART_val)
    ART_test = sparse.csr_matrix(ART_test)

    # song tim 내용 가져오기.
    tim_cate = []

    for i in tqdm_notebook(range(len(data_all))):
        tim = []
        songs = data_all.loc[i,'songs']

        for j in songs:
            tim.append(song.loc[j,'issue_date'])
        tim_cate.append(tim)


    data_all['plylst_times'] = tim_cate

    plylst_times = data_all['plylst_times']
    times_counter = Counter([tim for times in plylst_times for tim in times])
    times_dict = {x: times_counter[x] for x in times_counter}

    times_id_tid = dict()
    times_tid_id = dict()
    for i, t in enumerate(times_dict):
        times_id_tid[t] = i
        times_tid_id[i] = t
    n_times = len(times_dict)
    data_all['plylst_times_id'] = data_all['plylst_times'].map(lambda x: [times_id_tid.get(s) for s in x if times_id_tid.get(s) != None])

    tim_data_all = np.zeros((len(data_all),n_times))
    for i,index in enumerate(data_all.index):
        if i%10000 == 0:
            print(i)
        counter = Counter(data_all.loc[index]['plylst_times_id'])
        for (k,c) in counter.items():
            tim_data_all[i][k] = c

    tim_array = tim_data_all[:len(train)]
    tim_val = tim_data_all[len(train):len(train)+len(val)]
    tim_test = tim_data_all[len(train)+len(val):len(train)+len(val)+len(test)]

    # tim_data_all = sparse.csr_matrix(tim_data_all)
    del tim_data_all

    tim_array = sparse.csr_matrix(tim_array)
    tim_val = sparse.csr_matrix(tim_val)
    tim_test = sparse.csr_matrix(tim_test)

    #장르 대분류
    GEN_cate = []

    for i in tqdm_notebook(range(len(data_all))):
        GEN = []
        songs = data_all.loc[i,'songs']

        for j in songs:
            for k in song.loc[j,'song_gn_gnr_basket'] :
                GEN.append(k)
        GEN_cate.append(GEN)


    data_all['plylst_GENRE'] = GEN_cate

    plylst_GENRE = data_all['plylst_GENRE']
    GENRE_counter = Counter([GEN for GENRE in plylst_GENRE for GEN in GENRE])
    GENRE_dict = {x: GENRE_counter[x] for x in GENRE_counter}

    GENRE_id_tid = dict()
    GENRE_tid_id = dict()
    for i, t in enumerate(GENRE_dict):
        GENRE_id_tid[t] = i
        GENRE_tid_id[i] = t
    n_GENRE = len(GENRE_dict)
    data_all['plylst_GENRE_id'] = data_all['plylst_GENRE'].map(lambda x: [GENRE_id_tid.get(s) for s in x if GENRE_id_tid.get(s) != None])

    GEN_data_all = np.zeros((len(data_all),n_GENRE))
    for i,index in enumerate(data_all.index):
        if i%10000 == 0:
            print(i)
        counter = Counter(data_all.loc[index]['plylst_GENRE_id'])
        for (k,c) in counter.items():
            GEN_data_all[i][k] = c

    GEN_array = GEN_data_all[:len(train)]
    GEN_val = GEN_data_all[len(train):len(train)+len(val)]
    GEN_test = GEN_data_all[len(train)+len(val):len(train)+len(val)+len(test)]
    # GEN_data_all = sparse.csr_matrix(GEN_data_all)
    del GEN_data_all

    GEN_array = sparse.csr_matrix(GEN_array)
    GEN_val = sparse.csr_matrix(GEN_val)
    GEN_test = sparse.csr_matrix(GEN_test)

    content = data_all['plylst_title']
    if "{}.model".format(sp_total_model_path) not in os.listdir():
        makeSentencepieceModel(data_all,sp_total_model_path)
    sp = SentencePieceProcessor()
    sp.Load("{}.model".format(sp_total_model_path))

    cv = CountVectorizer(max_features=3000, tokenizer=sp.encode_as_pieces)
    content = data_all['plylst_title']
    tdm = cv.fit_transform(content)

    title_tdm = tdm.toarray()

    title_tr = title_tdm[:len(train)]
    title_va = title_tdm[len(train):len(train)+len(val)]
    title_ts = title_tdm[len(train)+len(val):len(train)+len(val)+len(test)]

    title_gnr = np.concatenate((gnr_array,title_tr),axis=1)
    val_title_gnr = np.concatenate((gnr_val,title_va),axis=1)
    test_title_gnr = np.concatenate((gnr_test,title_ts),axis=1)

    title_sp = sparse.csr_matrix(title_tdm)

    title_gnr = sparse.csr_matrix(title_gnr)
    val_title_gnr = sparse.csr_matrix(val_title_gnr)
    test_title_gnr = sparse.csr_matrix(test_title_gnr)

    title_gnr = vstack([title_gnr,val_title_gnr,test_title_gnr])
    song_sp = vstack([train_user_songs_A,val_user_songs_A,test_user_songs_A])
    tag_sp = vstack([train_user_tags_A,val_user_tags_A,test_user_tags_A])
    times_sp = vstack([tim_array,tim_val,tim_test])
    GEN_sp = vstack([GEN_array,GEN_val,GEN_test])


    ART_sp = vstack([ART_array,ART_val,ART_test])

    # song_sp_T = song_sp.T.tocsr()
    # tag_sp_T = tag_sp.T.tocsr()


    model_knn_song25 = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=25, n_jobs=-1)
    model_knn_tag25 = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=25, n_jobs=-1)
    model_knn_title25 = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=25, n_jobs=-1)
    model_knn_title_gnr25 = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=25, n_jobs=-1)
    model_knn_times25 = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=25, n_jobs=-1)
    model_knn_GEN25 = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=25, n_jobs=-1)
    model_knn_ART25 = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=25, n_jobs=-1)

    model_knn_song40 = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=40, n_jobs=-1)
    model_knn_tag40 = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=40, n_jobs=-1)
    model_knn_title40 = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=40, n_jobs=-1)
    model_knn_title_gnr40 = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=40, n_jobs=-1)
    model_knn_times40 = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=40, n_jobs=-1)
    model_knn_GEN40 = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=40, n_jobs=-1)
    model_knn_ART40 = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=40, n_jobs=-1)



    model_knn_song25.fit(song_sp)
    model_knn_tag25.fit(tag_sp)
    model_knn_title25.fit(title_sp)
    model_knn_title_gnr25.fit(title_gnr)
    model_knn_times25.fit(times_sp)
    model_knn_GEN25.fit(GEN_sp)
    model_knn_ART25.fit(ART_sp)

    model_knn_song40.fit(song_sp)
    model_knn_tag40.fit(tag_sp)
    model_knn_title40.fit(title_sp)
    model_knn_title_gnr40.fit(title_gnr)
    model_knn_times40.fit(times_sp)
    model_knn_GEN40.fit(GEN_sp)
    model_knn_ART40.fit(ART_sp)






    train.loc[:,'num_songs'] = train['songs'].map(len)
    train.loc[:,'num_tags'] = train['tags_id'].map(len)

    data_all = pd.concat([train,val,test])

    data_all.index = range(len(data_all))


    res = []
    for i in tqdm_notebook(range(len(test))):
        data = test.iloc[i]
        pid = i

        if len(data['songs']) >= 2 and len(data['tags_id']) >=2 :
            p = np.zeros((707989,1))
            p[data['songs']] = 1

            pp = np.zeros((n_tags,1))
            pp[data['tags_id']] = 1

            tra_song = data_all.iloc[model_knn_song25.kneighbors(p.T)[1][0]]
            row = np.repeat(range(25), tra_song['num_songs']) # User Index 별 노래 개수만큼 만듦
            col = [song for songs in tra_song['songs'] for song in songs] # Song dic number 추출
            dat = np.repeat(1, tra_song['num_songs'].sum()) # User별 Song이 있는 부분에 1을 넣기위해 1과 전체 노래 개수만큼 만듦
            tra_song_sp = spr.csr_matrix((dat, (row, col)), shape=(25, n_songs)) # csr_matrix 제작
            tra_song_sp_T = tra_song_sp.T.tocsr()

            tra_tag = data_all.iloc[model_knn_tag25.kneighbors(pp.T)[1][0]]
            row = np.repeat(range(25), tra_tag['num_tags'])
            col = [tag for tags in tra_tag['tags_id'] for tag in tags]
            dat = np.repeat(1, tra_tag['num_tags'].sum())
            tra_tag_sp = spr.csr_matrix((dat, (row, col)), shape=(25, n_tags))
            tra_tag_sp_T = tra_tag_sp.T.tocsr()

            tra_tim = times_sp[model_knn_times25.kneighbors(tim_test[i:(i+1)])[1][0]]
            tra_GEN = GEN_sp[model_knn_GEN25.kneighbors(GEN_test[i:(i+1)])[1][0]]
            tra_ART = ART_sp[model_knn_ART25.kneighbors(ART_test[i:(i+1)])[1][0]]
            tra_title_gnr = title_gnr[model_knn_title_gnr25.kneighbors(test_title_gnr[i:(i+1)])[1][0]]

            songs_already = data["songs"]
            tags_already = data["tags_id"]

            test_song = cosine_similarity(tra_song_sp,p.T)
            test_tag = cosine_similarity(tra_tag_sp,pp.T)

            test_tim = cosine_similarity(tra_tim,tim_test[i:(i+1)])
            test_GEN = cosine_similarity(tra_GEN,GEN_test[i:(i+1)])
            test_ART = cosine_similarity(tra_ART,ART_test[i:(i+1)])
            test_title_genre = cosine_similarity(tra_title_gnr,test_title_gnr[i:(i+1)])

            testi = test_song * test_tag * test_title_genre * test_tim * test_GEN * test_ART

            cand_song = tra_song_sp_T.dot(testi) # 행에는 노래 열에는 유저 정보 %*% 유사한 유저 -> 유사한 노래에 대하여 높은 값 나옴
            cand_song_idx = cand_song.reshape(-1).argsort()[-300:][::-1] # 값이 높은 상위 120개 노래 추출

            cand_song_idx = cand_song_idx[np.isin(cand_song_idx, songs_already) == False] # 중복제거
            cand1 = pd.DataFrame(cand_song).iloc[cand_song_idx].reset_index()
            ####### 40 ####################################################
            tra_song = data_all.iloc[model_knn_song40.kneighbors(p.T)[1][0]]
            row = np.repeat(range(40), tra_song['num_songs']) # User Index 별 노래 개수만큼 만듦
            col = [song for songs in tra_song['songs'] for song in songs] # Song dic number 추출
            dat = np.repeat(1, tra_song['num_songs'].sum()) # User별 Song이 있는 부분에 1을 넣기위해 1과 전체 노래 개수만큼 만듦
            tra_song_sp = spr.csr_matrix((dat, (row, col)), shape=(40, n_songs)) # csr_matrix 제작
            tra_song_sp_T = tra_song_sp.T.tocsr()

            tra_tag = data_all.iloc[model_knn_tag40.kneighbors(pp.T)[1][0]]
            row = np.repeat(range(40), tra_tag['num_tags'])
            col = [tag for tags in tra_tag['tags_id'] for tag in tags]
            dat = np.repeat(1, tra_tag['num_tags'].sum())
            tra_tag_sp = spr.csr_matrix((dat, (row, col)), shape=(40, n_tags))
            tra_tag_sp_T = tra_tag_sp.T.tocsr()

            tra_tim = times_sp[model_knn_times40.kneighbors(tim_test[i:(i+1)])[1][0]]
            tra_GEN = GEN_sp[model_knn_GEN40.kneighbors(GEN_test[i:(i+1)])[1][0]]
            tra_ART = ART_sp[model_knn_ART40.kneighbors(ART_test[i:(i+1)])[1][0]]
            tra_title_gnr = title_gnr[model_knn_title_gnr40.kneighbors(test_title_gnr[i:(i+1)])[1][0]]

            test_song = cosine_similarity(tra_song_sp,p.T)
            test_tag = cosine_similarity(tra_tag_sp,pp.T)

            test_tim = cosine_similarity(tra_tim,tim_test[i:(i+1)])
            test_GEN = cosine_similarity(tra_GEN,GEN_test[i:(i+1)])
            test_ART = cosine_similarity(tra_ART,ART_test[i:(i+1)])
            test_title_genre = cosine_similarity(tra_title_gnr,test_title_gnr[i:(i+1)])

            testi = test_song * test_tag * test_title_genre * test_tim * test_GEN * test_ART

            cand_song = tra_song_sp_T.dot(testi) # 행에는 노래 열에는 유저 정보 %*% 유사한 유저 -> 유사한 노래에 대하여 높은 값 나옴
            cand_song_idx = cand_song.reshape(-1).argsort()[-300:][::-1] # 값이 높은 상위 120개 노래 추출

            cand_song_idx = cand_song_idx[np.isin(cand_song_idx, songs_already) == False] # 중복제거
            cand2 = pd.DataFrame(cand_song).iloc[cand_song_idx].reset_index()

            cand_all = pd.merge(cand1,cand2,how='outer',on='index')
            cand_all = cand_all.fillna(0)
            cand_all['pred'] = (cand_all['0_x'] + cand_all['0_y'])/2
            cand_song_idx = list(cand_all.sort_values(by=['pred'],ascending=False)[:100]['index'])

    ######tag######
            cand_tag = tra_tag_sp_T.dot(testi) # 똑같은 작업 실시
            cand_tag_idx = cand_tag.reshape(-1).argsort()[-30:][::-1]

            cand_tag_idx = cand_tag_idx[np.isin(cand_tag_idx, tags_already) == False][:10]
            rec_tag_idx = [tag_tid_id[i] for i in cand_tag_idx]

            res.append({
                        "id": test.loc[pid,'id'],
                        "songs": cand_song_idx,
                        "tags": rec_tag_idx
                            })


        elif len(data['songs']) != 0:
            p = np.zeros((707989,1))
            p[data['songs']] = 1

            tra_song = data_all.iloc[model_knn_song25.kneighbors(p.T)[1][0]]
            row = np.repeat(range(25), tra_song['num_songs']) # User Index 별 노래 개수만큼 만듦
            col = [song for songs in tra_song['songs'] for song in songs] # Song dic number 추출
            dat = np.repeat(1, tra_song['num_songs'].sum()) # User별 Song이 있는 부분에 1을 넣기위해 1과 전체 노래 개수만큼 만듦
            tra_song_sp = spr.csr_matrix((dat, (row, col)), shape=(25, n_songs)) # csr_matrix 제작
            tra_song_sp_T = tra_song_sp.T.tocsr()

    #         tra_tag = data_all.iloc[model_knn_tag25.kneighbors(pp.T)[1][0]]
            row = np.repeat(range(25), tra_song['num_tags'])
            col = [tag for tags in tra_song['tags_id'] for tag in tags]
            dat = np.repeat(1, tra_song['num_tags'].sum())
            tra_tag_sp = spr.csr_matrix((dat, (row, col)), shape=(25, n_tags))
            tra_tag_sp_T = tra_tag_sp.T.tocsr()

            tra_tim = times_sp[model_knn_times25.kneighbors(tim_test[i:(i+1)])[1][0]]
            tra_GEN = GEN_sp[model_knn_GEN25.kneighbors(GEN_test[i:(i+1)])[1][0]]
            tra_ART = ART_sp[model_knn_ART25.kneighbors(ART_test[i:(i+1)])[1][0]]
            tra_title_gnr = title_gnr[model_knn_title_gnr25.kneighbors(test_title_gnr[i:(i+1)])[1][0]]

            songs_already = data["songs"]
            tags_already = data["tags_id"]

            test_song = cosine_similarity(tra_song_sp,p.T)
            test_tim = cosine_similarity(tra_tim,tim_test[i:(i+1)])
            test_GEN = cosine_similarity(tra_GEN,GEN_test[i:(i+1)])
            test_ART = cosine_similarity(tra_ART,ART_test[i:(i+1)])
            test_title_genre = cosine_similarity(tra_title_gnr,test_title_gnr[i:(i+1)])

            testi = test_song*test_title_genre*test_tim*test_GEN * test_ART

            cand_song = tra_song_sp_T.dot(testi) # 행에는 노래 열에는 유저 정보 %*% 유사한 유저 -> 유사한 노래에 대하여 높은 값 나옴
            cand_song_idx = cand_song.reshape(-1).argsort()[-300:][::-1] # 값이 높은 상위 120개 노래 추출

            cand_song_idx = cand_song_idx[np.isin(cand_song_idx, songs_already) == False] # 중복제거
            cand1 = pd.DataFrame(cand_song).iloc[cand_song_idx].reset_index()
            ####### 40 ####################################################
            tra_song = data_all.iloc[model_knn_song40.kneighbors(p.T)[1][0]]
            row = np.repeat(range(40), tra_song['num_songs']) # User Index 별 노래 개수만큼 만듦
            col = [song for songs in tra_song['songs'] for song in songs] # Song dic number 추출
            dat = np.repeat(1, tra_song['num_songs'].sum()) # User별 Song이 있는 부분에 1을 넣기위해 1과 전체 노래 개수만큼 만듦
            tra_song_sp = spr.csr_matrix((dat, (row, col)), shape=(40, n_songs)) # csr_matrix 제작
            tra_song_sp_T = tra_song_sp.T.tocsr()

            row = np.repeat(range(40), tra_song['num_tags'])
            col = [tag for tags in tra_song['tags_id'] for tag in tags]
            dat = np.repeat(1, tra_song['num_tags'].sum())
            tra_tag_sp = spr.csr_matrix((dat, (row, col)), shape=(40, n_tags))
            tra_tag_sp_T = tra_tag_sp.T.tocsr()

            tra_tim = times_sp[model_knn_times40.kneighbors(tim_test[i:(i+1)])[1][0]]
            tra_GEN = GEN_sp[model_knn_GEN40.kneighbors(GEN_test[i:(i+1)])[1][0]]
            tra_ART = ART_sp[model_knn_ART40.kneighbors(ART_test[i:(i+1)])[1][0]]
            tra_title_gnr = title_gnr[model_knn_title_gnr40.kneighbors(test_title_gnr[i:(i+1)])[1][0]]

            test_song = cosine_similarity(tra_song_sp,p.T)
            test_tim = cosine_similarity(tra_tim,tim_test[i:(i+1)])
            test_GEN = cosine_similarity(tra_GEN,GEN_test[i:(i+1)])
            test_ART = cosine_similarity(tra_ART,ART_test[i:(i+1)])
            test_title_genre = cosine_similarity(tra_title_gnr,test_title_gnr[i:(i+1)])

            testi = test_song * test_title_genre * test_tim * test_GEN * test_ART

            cand_song = tra_song_sp_T.dot(testi) # 행에는 노래 열에는 유저 정보 %*% 유사한 유저 -> 유사한 노래에 대하여 높은 값 나옴
            cand_song_idx = cand_song.reshape(-1).argsort()[-300:][::-1] # 값이 높은 상위 120개 노래 추출

            cand_song_idx = cand_song_idx[np.isin(cand_song_idx, songs_already) == False] # 중복제거
            cand2 = pd.DataFrame(cand_song).iloc[cand_song_idx].reset_index()

            cand_all = pd.merge(cand1,cand2,how='outer',on='index')
            cand_all = cand_all.fillna(0)
            cand_all['pred'] = (cand_all['0_x'] + cand_all['0_y'])/2
            cand_song_idx = list(cand_all.sort_values(by=['pred'],ascending=False)[:100]['index'])

    #######tag########
            cand_tag = tra_tag_sp_T.dot(testi) # 똑같은 작업 실시
            cand_tag_idx = cand_tag.reshape(-1).argsort()[-30:][::-1]

            cand_tag_idx = cand_tag_idx[np.isin(cand_tag_idx, tags_already) == False][:10]
            rec_tag_idx = [tag_tid_id[i] for i in cand_tag_idx]

            res.append({
                        "id": test.loc[pid,'id'],
                        "songs": cand_song_idx,
                        "tags": rec_tag_idx
                        })


        elif len(data['tags_id']) !=0:
            p = np.zeros((n_tags,1))
            p[data['tags_id']] = 1

            tra_tag = data_all.iloc[model_knn_tag25.kneighbors(p.T)[1][0]]
            row = np.repeat(range(25), tra_tag['num_songs']) # User Index 별 노래 개수만큼 만듦
            col = [song for songs in tra_tag['songs'] for song in songs] # Song dic number 추출
            dat = np.repeat(1, tra_tag['num_songs'].sum()) # User별 Song이 있는 부분에 1을 넣기위해 1과 전체 노래 개수만큼 만듦
            tra_song_sp = spr.csr_matrix((dat, (row, col)), shape=(25, n_songs)) # csr_matrix 제작
            tra_song_sp_T = tra_song_sp.T.tocsr()

            row = np.repeat(range(25), tra_tag['num_tags'])
            col = [tag for tags in tra_tag['tags_id'] for tag in tags]
            dat = np.repeat(1, tra_tag['num_tags'].sum())
            tra_tag_sp = spr.csr_matrix((dat, (row, col)), shape=(25, n_tags))
            tra_tag_sp_T = tra_tag_sp.T.tocsr()


            songs_already = data["songs"]
            tags_already = data["tags_id"]

            testi = cosine_similarity(tra_tag_sp,pp.T)

            if len(data['plylst_title']) != 0 :
                tra_title_gnr = title_tdm[model_knn_title25.kneighbors(title_ts[i:(i+1)])[1][0]]
                testi_title = cosine_similarity(tra_title_gnr,title_ts[i:(i+1)])
                testi = testi * testi_title

            cand_song = tra_song_sp_T.dot(testi) # 행에는 노래 열에는 유저 정보 %*% 유사한 유저 -> 유사한 노래에 대하여 높은 값 나옴
            cand_song_idx = cand_song.reshape(-1).argsort()[-300:][::-1] # 값이 높은 상위 120개 노래 추출

            cand_song_idx = cand_song_idx[np.isin(cand_song_idx, songs_already) == False][:100]  # 중복되는 노래 있는지 확인하고 100개 추출

            cand_tag = tra_tag_sp_T.dot(testi) # 똑같은 작업 실시
            cand_tag_idx = cand_tag.reshape(-1).argsort()[-30:][::-1]

            cand_tag_idx = cand_tag_idx[np.isin(cand_tag_idx, tags_already) == False][:10]
            rec_tag_idx = [tag_tid_id[i] for i in cand_tag_idx]

            res.append({
                        "id": test.loc[pid,'id'],
                        "songs": list(cand_song_idx),
                        "tags": rec_tag_idx
                        })

        else :
            cand_song = []
            for li in data_all.iloc[model_knn_title25.kneighbors(title_ts[i:(i+1)])[1][0]].songs.to_list():
                for j in li:
                    cand_song.append(j)

            cand_tag = []
            for li in data_all.iloc[model_knn_title25.kneighbors(title_ts[i:(i+1)])[1][0]].tags.to_list():
                for j in li:
                    cand_tag.append(j)

            cand_song_idx = list(pd.DataFrame(cand_song)[0].value_counts()[:100].index)
            rec_tag_idx = list(pd.DataFrame(cand_tag)[0].value_counts()[:10].index)

            res.append({
                        "id": test.loc[pid,'id'],
                        "songs": cand_song_idx,
                        "tags": rec_tag_idx
                        })

    for i in range(len(res)):
        if len(res[i]['songs']) != 100:
            print('song 에서 {}번째 오류 발생'.format(i))

        if len(res[i]['tags']) != 10:
            print('tag 에서 {}번째 오류 발생'.format(i))

    rec = []
    for i in range(len(res)):
        rec.append({
                        "id": res[i]['id'],
                        "songs": list(res[i]['songs']),
                        "tags": res[i]['tags']
                        })

    result1 = pd.DataFrame(rec)

    model_knn_song = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=50, n_jobs=-1)
    model_knn_tag = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=50, n_jobs=-1)
    model_knn_title = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=50, n_jobs=-1)
    model_knn_title_gnr = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=50, n_jobs=-1)
    model_knn_times = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=50, n_jobs=-1)
    model_knn_GEN = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=50, n_jobs=-1)
    model_knn_ART = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=50, n_jobs=-1)

    model_knn_song.fit(song_sp)
    model_knn_tag.fit(tag_sp)
    model_knn_title.fit(title_sp)
    model_knn_title_gnr.fit(title_gnr)
    model_knn_times.fit(times_sp)
    model_knn_GEN.fit(GEN_sp)
    model_knn_ART.fit(ART_sp)

    res2 = []
    for i in tqdm_notebook([1960, 6361, 8705, 9310, 10498]):
        data = test.iloc[i]
        pid = i

        if len(data['songs']) != 0 and len(data['tags_id']) != 0:
            p = np.zeros((707989,1))
            p[data['songs']] = 1

            pp = np.zeros((n_tags,1))
            pp[data['tags_id']] = 1

            tra_song = data_all.iloc[model_knn_song.kneighbors(p.T)[1][0]]
            row = np.repeat(range(50), tra_song['num_songs']) # User Index 별 노래 개수만큼 만듦
            col = [song for songs in tra_song['songs'] for song in songs] # Song dic number 추출
            dat = np.repeat(1, tra_song['num_songs'].sum()) # User별 Song이 있는 부분에 1을 넣기위해 1과 전체 노래 개수만큼 만듦
            tra_song_sp = spr.csr_matrix((dat, (row, col)), shape=(50, n_songs)) # csr_matrix 제작
            tra_song_sp_T = tra_song_sp.T.tocsr()

            tra_tag = data_all.iloc[model_knn_tag.kneighbors(pp.T)[1][0]]
            row = np.repeat(range(50), tra_tag['num_tags'])
            col = [tag for tags in tra_tag['tags_id'] for tag in tags]
            dat = np.repeat(1, tra_tag['num_tags'].sum())
            tra_tag_sp = spr.csr_matrix((dat, (row, col)), shape=(50, n_tags))
            tra_tag_sp_T = tra_tag_sp.T.tocsr()

            tra_tim = times_sp[model_knn_times.kneighbors(tim_test[i:(i+1)])[1][0]]
            tra_GEN = GEN_sp[model_knn_GEN.kneighbors(GEN_test[i:(i+1)])[1][0]]
            tra_title_gnr = title_gnr[model_knn_title_gnr.kneighbors(test_title_gnr[i:(i+1)])[1][0]]

            songs_already = data["songs"]
            tags_already = data["tags_id"]

            test_song = cosine_similarity(tra_song_sp,p.T)
            test_tag = cosine_similarity(tra_tag_sp,pp.T)

            test_tim = cosine_similarity(tra_tim,tim_test[i:(i+1)])
            test_GEN = cosine_similarity(tra_GEN,GEN_test[i:(i+1)])
            test_title_genre = cosine_similarity(tra_title_gnr,test_title_gnr[i:(i+1)])

            testi = test_song * test_tag * test_title_genre * test_GEN

            cand_song = tra_song_sp_T.dot(testi) # 행에는 노래 열에는 유저 정보 %*% 유사한 유저 -> 유사한 노래에 대하여 높은 값 나옴
            cand_song_idx = cand_song.reshape(-1).argsort()[-300:][::-1] # 값이 높은 상위 120개 노래 추출

            cand_song_idx = cand_song_idx[np.isin(cand_song_idx, songs_already) == False][:100]  # 중복되는 노래 있는지 확인하고 100개 추출

            cand_tag = tra_tag_sp_T.dot(testi) # 똑같은 작업 실시
            cand_tag_idx = cand_tag.reshape(-1).argsort()[-30:][::-1]

            cand_tag_idx = cand_tag_idx[np.isin(cand_tag_idx, tags_already) == False][:10]
            rec_tag_idx = [tag_tid_id[i] for i in cand_tag_idx]

            res2.append({
                        "id": test.loc[pid,'id'],
                        "songs": cand_song_idx,
                        "tags": rec_tag_idx
                            })


        elif len(data['songs']) != 0:
            p = np.zeros((707989,1))
            p[data['songs']] = 1

            tra_song = data_all.iloc[model_knn_song.kneighbors(p.T)[1][0]]
            row = np.repeat(range(50), tra_song['num_songs']) # User Index 별 노래 개수만큼 만듦
            col = [song for songs in tra_song['songs'] for song in songs] # Song dic number 추출
            dat = np.repeat(1, tra_song['num_songs'].sum()) # User별 Song이 있는 부분에 1을 넣기위해 1과 전체 노래 개수만큼 만듦
            tra_song_sp = spr.csr_matrix((dat, (row, col)), shape=(50, n_songs)) # csr_matrix 제작
            tra_song_sp_T = tra_song_sp.T.tocsr()

            row = np.repeat(range(50), tra_song['num_tags'])
            col = [tag for tags in tra_song['tags_id'] for tag in tags]
            dat = np.repeat(1, tra_song['num_tags'].sum())
            tra_tag_sp = spr.csr_matrix((dat, (row, col)), shape=(50, n_tags))
            tra_tag_sp_T = tra_tag_sp.T.tocsr()

            songs_already = data["songs"]
            tags_already = data["tags_id"]

            tra_tim = times_sp[model_knn_times.kneighbors(tim_test[i:(i+1)])[1][0]]
            tra_GEN = GEN_sp[model_knn_GEN.kneighbors(GEN_test[i:(i+1)])[1][0]]
            tra_title_gnr = title_gnr[model_knn_title_gnr.kneighbors(test_title_gnr[i:(i+1)])[1][0]]

            test_song = cosine_similarity(tra_song_sp,p.T)

            test_tim = cosine_similarity(tra_tim,tim_test[i:(i+1)])
            test_GEN = cosine_similarity(tra_GEN,GEN_test[i:(i+1)])
            test_title_genre = cosine_similarity(tra_title_gnr,test_title_gnr[i:(i+1)])
            testi = test_song*test_title_genre*test_tim*test_GEN

            cand_song = tra_song_sp_T.dot(testi) # 행에는 노래 열에는 유저 정보 %*% 유사한 유저 -> 유사한 노래에 대하여 높은 값 나옴
            cand_song_idx = cand_song.reshape(-1).argsort()[-200:][::-1] # 값이 높은 상위 120개 노래 추출

            cand_song_idx = cand_song_idx[np.isin(cand_song_idx, songs_already) == False][:100]  # 중복되는 노래 있는지 확인하고 100개 추출

            cand_tag = tra_tag_sp_T.dot(testi) # 똑같은 작업 실시
            cand_tag_idx = cand_tag.reshape(-1).argsort()[-30:][::-1]

            cand_tag_idx = cand_tag_idx[np.isin(cand_tag_idx, tags_already) == False][:10]
            rec_tag_idx = [tag_tid_id[i] for i in cand_tag_idx]

            res2.append({
                        "id": test.loc[pid,'id'],
                        "songs": cand_song_idx,
                        "tags": rec_tag_idx
                        })

        elif len(data['tags_id']) !=0:
            p = np.zeros((n_tags,1))
            p[data['tags_id']] = 1

            tra_tag = data_all.iloc[model_knn_tag.kneighbors(p.T)[1][0]]
            row = np.repeat(range(50), tra_tag['num_songs']) # User Index 별 노래 개수만큼 만듦
            col = [song for songs in tra_tag['songs'] for song in songs] # Song dic number 추출
            dat = np.repeat(1, tra_tag['num_songs'].sum()) # User별 Song이 있는 부분에 1을 넣기위해 1과 전체 노래 개수만큼 만듦
            tra_song_sp = spr.csr_matrix((dat, (row, col)), shape=(50, n_songs)) # csr_matrix 제작
            tra_song_sp_T = tra_song_sp.T.tocsr()

            row = np.repeat(range(50), tra_tag['num_tags'])
            col = [tag for tags in tra_tag['tags_id'] for tag in tags]
            dat = np.repeat(1, tra_tag['num_tags'].sum())
            tra_tag_sp = spr.csr_matrix((dat, (row, col)), shape=(50, n_tags))
            tra_tag_sp_T = tra_tag_sp.T.tocsr()


            songs_already = data["songs"]
            tags_already = data["tags_id"]

            testi = cosine_similarity(tra_tag_sp,pp.T)

            if len(data['plylst_title']) != 0 :
                tra_title_gnr = title_tdm[model_knn_title.kneighbors(title_ts[i:(i+1)])[1][0]]
                testi_title = cosine_similarity(tra_title_gnr,title_ts[i:(i+1)])
                testi = testi * testi_title

            cand_song = tra_song_sp_T.dot(testi) # 행에는 노래 열에는 유저 정보 %*% 유사한 유저 -> 유사한 노래에 대하여 높은 값 나옴
            cand_song_idx = cand_song.reshape(-1).argsort()[-300:][::-1] # 값이 높은 상위 120개 노래 추출

            cand_song_idx = cand_song_idx[np.isin(cand_song_idx, songs_already) == False][:100]  # 중복되는 노래 있는지 확인하고 100개 추출

            cand_tag = tra_tag_sp_T.dot(testi) # 똑같은 작업 실시
            cand_tag_idx = cand_tag.reshape(-1).argsort()[-30:][::-1]

            cand_tag_idx = cand_tag_idx[np.isin(cand_tag_idx, tags_already) == False][:10]
            rec_tag_idx = [tag_tid_id[i] for i in cand_tag_idx]

            res2.append({
                        "id": test.loc[pid,'id'],
                        "songs": cand_song_idx,
                        "tags": rec_tag_idx
                        })

        else:
            cand_song = []
            for li in data_all.iloc[model_knn_title.kneighbors(title_ts[i:(i+1)])[1][0]].songs.to_list():
                for j in li:
                    cand_song.append(j)

            cand_tag = []
            for li in data_all.iloc[model_knn_title.kneighbors(title_ts[i:(i+1)])[1][0]].tags.to_list():
                for j in li:
                    cand_tag.append(j)

            cand_song_idx = list(pd.DataFrame(cand_song)[0].value_counts()[:100].index)
            rec_tag_idx = list(pd.DataFrame(cand_tag)[0].value_counts()[:10].index)

            res2.append({
                        "id": test.loc[pid,'id'],
                        "songs": cand_song_idx,
                        "tags": rec_tag_idx
                        })


    pd.DataFrame(res2)

    rec2 = []
    for i in range(len(res2)):
        rec2.append({
                        "id": res2[i]['id'],
                        "songs": list(res2[i]['songs']),
                        "tags": res2[i]['tags']
                        })

    result2 = pd.DataFrame(rec2)['songs']

    n_index = [10498,6361,1960,8705,9310]

    result2.index = n_index

    result1.loc[n_index,'songs'] = result2

    result1['songs'].apply(len).sort_values()
    #그럼에도 채워지지 않은 6361에 대해서 상위 100곡 추천
    s = []
    for song in train.songs.tolist():
        s += song
    r1 = dict(Counter(s))

    r_song = sorted(r1.items(), key=lambda x: -x[1])
    r_song_top = r_song[:100] # 몇 곡 할지도 정해야 함

    list_song = list(dict(r_song_top).keys())
    len(list_song)

    sub= []
    for j in range(len(result1)) :
        sub.append(result1.loc[j].to_dict())

    sub[6361]['songs'] = list_song

    pd.DataFrame(sub)['songs'].apply(len).sort_values()
    write_json(sub,'final_songs.json')
    return sub

if __name__ == '__main__':

    _data = Dataset()

    #pre_tag.run(_data.test,_data.n_songs,_data.n_tags,_data.spr_list,_data.tag_tid_id)
    final_tags = word2vec_for_tag.run(_data.total,_data.test)

    final_songs = song_inference()
    result = []
    for f_songs, f_tags in zip(final_songs,final_tags):
        result.append({
            'id':f_songs['id'],
            'songs':f_songs['songs'],
            'tags':f_tags['tags']
        })
    write_json(result, 'results.json')
