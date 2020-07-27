import io
import os
import json
import distutils.dir_util
import numpy as np
from sentencepiece import SentencePieceTrainer

def write_json(data, fname):
    def _conv(o):
        if isinstance(o, (np.int64, np.int32)):
            return int(o)
        raise TypeError

    parent = os.path.dirname(fname)
    distutils.dir_util.mkpath("./arena_data/" + parent)
    with io.open("./arena_data/" + fname, "w", encoding="utf-8") as f:
        json_str = json.dumps(data, ensure_ascii=False, default=_conv)
        f.write(json_str)

def remove_seen(seen, l):
    seen = set(seen)
    return [x for x in l if not (x in seen)]

#song 예측에는 -> train + val + test
#tag 예측에는 -> train
def makeSentencepieceModel(df,fname):
    content = df['plylst_title']
    with open('{}.txt'.format(fname), 'w', encoding='utf8') as f:
        f.write('\n'.join(content))
    SentencePieceTrainer.Train('--input={}.txt --model_prefix={} --vocab_size=3000'.format(fname,fname))
