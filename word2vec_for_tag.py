import json
import numpy as np
import pandas as pd
import io,os,json,re
import distutils.dir_util
import util
from collections import Counter
from gensim.models import Word2Vec
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors
from util import write_json,remove_seen
import train

#total_concat : pd.DataFrame
def run(total_concat, apply_data):
    total_concat['id'] = total_concat['id'].astype(str)
    c = Counter()
    for i in total_concat['tags']:
        c.update(i)
    tag_list = list(map(lambda y:y[0],(filter(lambda x: x[1]>5,c.items()))))

    p = re.compile('|'.join(tag_list))

    total_concat['tag_in_title']=total_concat['plylst_title'].apply(lambda x:p.findall(x))

    data = []
    for i in total_concat.index:
        temp = total_concat.iloc[i]
        data.append({
            'id':temp['id'],
            'songs':temp['songs'],
            'tags':temp['tags'],
            'tag_in_title':temp['tag_in_title']
        })
    song_dic = {}
    tag_dic = {}
    for q in data:
        song_dic[q['id']] = q['songs']
        tag_dic[q['id']] = q['tags']
    total = list(map(lambda x: list(map(str,x['songs'])) + x['tags']+x['tag_in_title'], data))
    total = [x for x in total if len(x)>1]

    print("start training item2Vec")
    size = 300
    if 'item2vec.model' in os.listdir():
        w2v_model = Word2Vec.load('item2vec.model')
    else:
        w2v_model = train.item2vec(total,size=size)
    print("done. \n")
    p2v_model = WordEmbeddingsKeyedVectors(size)
    ID = []
    vec = []
    for q in data:
        tmp_vec = 0
        for song in list(map(str,q['songs'])) + q['tags']+q['tag_in_title']:
            try:
                tmp_vec += w2v_model.wv.get_vector(song)
            except KeyError:
                pass
        if type(tmp_vec)!=int:
            ID.append(str(q['id']))
            vec.append(tmp_vec)
    p2v_model.add(ID, vec)


    with open("./arena_data/pre_tag.json", encoding="utf-8") as f:
        our_best = json.load(f)

    not_in = 0
    answers = []
    for i,q in enumerate(apply_data.index):
        q = apply_data.loc[q]
        try:
            most_id = [x[0] for x in p2v_model.most_similar(str(q['id']), topn=200)]
            get_song = []
            get_tag = []
            for ID in most_id:
                get_song += song_dic[ID]
                get_tag += tag_dic[ID]
            get_song = list(pd.value_counts(get_song)[:300].index)
            get_tag = list(pd.value_counts(get_tag)[:30].index)

            output_song = remove_seen(q["songs"], get_song)[:100]
            output_tag = remove_seen(q["tags"], get_tag)[:10]

            answers.append({
                "id": q["id"],
                "songs": output_song,
                "tags": output_tag,
            })
        except KeyError:
            not_in += 1
            answers.append({
                "id": our_best[i]["id"],
                "songs": our_best[i]['songs'],
                "tags": our_best[i]["tags"],
            })

    for n, q in enumerate(answers):
        if len(q['songs'])!=100:
            answers[n]['songs'] += remove_seen(q['songs'], our_best[n]['songs'])[:100-len(q['songs'])]
        if len(q['tags'])!=10:
            answers[n]['tags'] += remove_seen(q['tags'], our_best[n]['tags'])[:10-len(q['tags'])]
    write_json(answers,'final_tags.json')
    return answers
