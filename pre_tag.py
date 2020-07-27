import pandas as pd
import numpy as np
import time
import scipy.sparse as spr
from util import write_json
from sklearn.metrics.pairwise import cosine_similarity

def run(test,n_songs,n_tags,spr_list,tag_tid_id):
    start = time.time()
    train_user_songs_A,train_user_tags_A,\
        test_title,title_sp,gnr_sp,test_gnr_sp,\
        title_gnr,test_title_gnr = spr_list

    res = []
    for i in range(len(test)):
        dat = test.iloc[i]
        pid = i
        songs_already = dat["songs"]
        tags_already = dat["tags_id"]

        if len(dat['songs']) != 0 and len(dat['tags_id']) != 0:
            p = np.zeros((n_songs,1))
            p[dat['songs']] = 1
            val_song = cosine_similarity(train_user_songs_A,p.T)

            pp = np.zeros((n_tags,1))
            pp[dat['tags_id']] = 1
            val_tag = cosine_similarity(train_user_tags_A,pp.T)

            val_title_genre = cosine_similarity(title_gnr,test_title_gnr[i:(i+1)])
            val = val_song * val_tag * val_title_genre


        elif len(dat['songs']) != 0:
            p = np.zeros((n_songs,1))
            p[dat['songs']] = 1
            val_song = cosine_similarity(train_user_songs_A,p.T)

            val_title_genre = cosine_similarity(title_gnr,test_title_gnr[i:(i+1)])
            val = val_song * val_title_genre


        elif len(dat['tags_id']) !=0:
            p = np.zeros((n_tags,1))
            p[dat['tags_id']] = 1

            val = cosine_similarity(train_user_tags_A,p.T)

            if len(dat['plylst_title']) != 0 :
                val_title = cosine_similarity(title_sp,test_title[i:(i+1)])
                val = val * val_title

        else:
            val = cosine_similarity(title_sp,test_title[i:(i+1)])

        cand_song = train_user_songs_A.T.tocsr().dot(val) # 행에는 노래 열에는 유저 정보 %*% 유사한 유저 -> 유사한 노래에 대하여 높은 값 나옴
        cand_song_idx = cand_song.reshape(-1).argsort()[-300:][::-1] # 값이 높은 상위 150개 노래 추출
        cand_song_idx = cand_song_idx[np.isin(cand_song_idx, songs_already) == False][:100]  # 중복되는 노래 있는지 확인하고 100개 추출

        cand_tag = train_user_tags_A.T.tocsr().dot(val) # 똑같은 작업 실시
        cand_tag_idx = cand_tag.reshape(-1).argsort()[-30:][::-1]
        cand_tag_idx = cand_tag_idx[np.isin(cand_tag_idx, tags_already) == False][:10]
        rec_tag_idx = [tag_tid_id[i] for i in cand_tag_idx]

        res.append({
                    "id": test.loc[pid,'id'],
                    "songs": list(cand_song_idx),
                    "tags": rec_tag_idx
                    })

        if i % 1000 == 0:
            print("{} time :".format(i), time.time() - start)

    write_json(res, "./dataset/pre_tag.json")
