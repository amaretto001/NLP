# -*- coding: utf-8 -*-

import MeCab
from gensim.models import word2vec
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import re
import matplotlib.pyplot as plt

tagger = MeCab.Tagger('-d /usr/local/lib/mecab/dic/ipadic')
tagger.parse("")
path = "C:/Users/user/python/"

##############################################################################

def preprocessing(sentence):
    return sentence.rstrip()

def extract_noun_by_parse(path):
    nouns = []
    with open(path + "rvwlist.txt", "r", encoding='UTF-8') as fd:#開封
        for sentence in map(preprocessing, fd):
            sentence = re.sub(r'[0123456789０１２３４５６７８９！＠＃＄％＾＆\-|\\＊\“（）＿_■×※⇒—●(：〜＋=)／*&^%$#@!~`){}…\[\]\"\'\”:;<>?＜＞？、。・,./『』【】「」→←○]+', "", sentence)
            for chunk in tagger.parse(sentence).splitlines()[:-1]:
                chunk = chunk.split('\t')
                surface = chunk[0]
                feature = chunk[3]
                if feature.startswith(('名詞', '形容詞')):
                    nouns.append(surface)
    return nouns

##############################################################################

nouns = extract_noun_by_parse(path)

wakati_file = 'rvwlist.wakati'
with open(wakati_file, 'w', encoding='utf-8') as fp:
    fp.write("\n".join(nouns))

data = word2vec.LineSentence(wakati_file)
model = word2vec.Word2Vec(data,
                          size= 300, window=20, hs=1, min_count=1, sg=5)
model.save('rvwlist.model')

##############################################################################

word2vec_model=model

skip=0
limit=241 

vocab = word2vec_model.wv.vocab
emb_tuple = tuple([word2vec_model[v] for v in vocab])
X = np.vstack(emb_tuple)

tsne_model = TSNE(n_components=2, random_state=0,verbose=2)
np.set_printoptions(suppress=True)
tsne_model.fit_transform(X)

plain_tsne = pd.DataFrame(tsne_model.embedding_[skip:limit, 0],columns = ["x"])
plain_tsne["y"] = pd.DataFrame(tsne_model.embedding_[skip:limit, 1])
plain_tsne["word"] = list(vocab)[skip:limit]

fig, ax = plt.subplots()
plain_tsne.plot(x="x",y="y",kind='scatter',figsize=(20, 20),ax=ax)

for k, v in plain_tsne.iterrows():
    ax.annotate(v[2],xy=(v[0],v[1]),size=10)
