import gensim
import nltk
import pandas as pd
import re
import urllib.request
import numpy as np
from dostoevsky.tokenization import RegexTokenizer
from dostoevsky.models import FastTextSocialNetworkModel
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk import FreqDist
from sklearn.manifold import TSNE
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
from PIL import Image
from io import BytesIO
import base64
import gensim.downloader as api

from functools import lru_cache
from gensim.models.word2vec import Word2Vec
from gensim.models import *
from gensim import corpora
from gensim import similarities
from string import punctuation
from pymorphy2 import MorphAnalyzer

from nltk import FreqDist
from tqdm.notebook import tqdm
from sklearn.manifold import TSNE


nltk.download('stopwords')
nltk.download('punkt')


def preproc(texts):
    pymorphy2_analyzer = MorphAnalyzer()
    stopwords = set(nltk.corpus.stopwords.words('russian'))
    digits = '0123456789'
    texts = re.sub('[{}]'.format(punctuation + digits), '', texts)
    texts = [pymorphy2_analyzer.parse(elem)[0].normal_form for elem in texts.split()]
    return [word for word in texts if word not in stopwords]

def getSent(surname_user, df):
  neu = list(df['neutral'].values)
  neg = list(df['negative'].values)
  pos = list(df['positive'].values)
  res_neu = []
  res_neg = []
  res_pos = []
  b = list(df['name'].values)
  for i in range(len(b)):
    if b[i] == surname_user:
      res_neu.append(neu[i])
      res_neg.append(neg[i])
      res_pos.append(pos[i])
  return round(sum(res_neu)/len(res_neu),2), round(sum(res_neg)/len(res_neg), 2), round(sum(res_pos)/len(res_pos), 2)

def remove_stopwords(text):
    customstopwords = ['это', 'которые', 'просто', 'почему', 'который']
    mystopwords = stopwords.words('russian') + ['br'] + customstopwords
    try:
        return u" ".join([token for token in nltk.word_tokenize(text) if not token in mystopwords])
    except:
        raise
        return u""

def getText(surname_user, df):
  result = []
  a = list(df['text'].values)
  b = list(df['name'].values)
  for i in range(len(a)):
    if b[i] == surname_user:
      result.append(a[i])
  return result

def im_2_b64(image):
    buff = BytesIO()
    image.save(buff, format="JPEG")
    img_str = base64.b64encode(buff.getvalue())
    return img_str


async def get_super_info(user_info):
    '''user_info={
      "sub": "d276ec5a-33fe-4526-aa51-60f9c2340e5d",
      "patronymic": "Михайлович",
      "birthdate": "9/17/1985",
      "mobile": "78563741122",
      "given_name": "Александр",
      "family_name": "Овечкин",
      "email": "ovechkin.a@yandex.ru",
      "username": "ovechkin"
    }'''
    surname_user = user_info['family_name'].lower()
    df = pd.read_csv('./data/names.csv')

    df.text = df.text.apply(remove_stopwords)
    df["name"] = np.nan
    #names = ['медведева', 'акинфеев', 'овечкин', 'оганджанянц', 'бухман', 'дудь', 'картынник', 'тиньков', 'наумова', 'перельман', 'абрамович', 'андреев', 'айтиборода']
    for index, row in df.iterrows():
        if str(surname_user) in str(df['text'][index]):
            df['name'][index] = str(surname_user)
            #break
    long_string = ','.join(list(getText(surname_user=surname_user, df=df)))
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
    wordcloud.generate(long_string)
    image = wordcloud.to_image()
    #print(image)

    tokenizer = RegexTokenizer()
    FastTextSocialNetworkModel.MODEL_PATH = './data/fasttext-social-network-model.bin'
    model = FastTextSocialNetworkModel(tokenizer=tokenizer)

    list_of_texts = df.text

    sentiment_list = []
    results = model.predict(list_of_texts, k=2)
    for sentiment in results:
        sentiment_list.append(sentiment)

    neutral_list = []
    negative_list = []
    positive_list = []
    speech_list = []
    skip_list = []
    for sentiment in sentiment_list:
        neutral = sentiment.get('neutral')
        negative = sentiment.get('negative')
        positive = sentiment.get('positive')
        if neutral is None:
            neutral_list.append(0)
            neutral = 0
        else:
            neutral_list.append(neutral)
        if negative is None:
            negative_list.append(0)
            negative = 0
        else:
            negative_list.append(negative)
        if positive is None:
            positive_list.append(1-neutral-negative)
        else:
            positive_list.append(sentiment.get('positive'))
    df['neutral'] = neutral_list
    df['negative'] = negative_list
    df['positive'] = positive_list
    neutr, negat, posit = getSent(surname_user=surname_user,df=df)
    user_info['netral'] = neutr
    user_info['negat'] =  negat
    user_info['posit'] = posit
    #print(user_info)
    user_info['image_wordcount'] = im_2_b64(image)
    image_user = Image.open('./data/'+user_info['username']+'.jpg')
    user_info['image_user'] = im_2_b64(image_user)

    texts = df[df.name == surname_user]['text'].str

    n = 100000
    corpus = texts[:n].map(preproc)
    model_1 = Word2Vec(corpus, min_count = 0, workers=8)

    top_words = []

    fd = FreqDist()
    for words in tqdm(corpus):
        fd.update(words)

    for w in fd.most_common(1000):
        top_words.append(w[0])

    top_words_vec = model_1.wv[top_words]

    print('Making dictionary...')
    dictionary = corpora.Dictionary(corpus)
    print('Original: {}'.format(dictionary))
    dictionary.filter_extremes(no_below = 5, no_above = 0.9, keep_n=None)
    dictionary.save('polkrug.dict')
    print('Filtered: {}'.format(dictionary))

    print('Vectorizing corpus...')
    corpus = [dictionary.doc2bow(text) for text in corpus]
    corpora.MmCorpus.serialize('polkrug.model', corpus)

    lda = ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=10)
    lda.show_topics(num_topics=10, num_words=10, formatted=False)

    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

    cloud = WordCloud(stopwords=stopwords,
                      background_color='white',
                      width=2500,
                      height=1800,
                      max_words=10,
                      colormap='tab10',
                      color_func=lambda *args, **kwargs: cols[i],
                      prefer_horizontal=1.0)

    topics = lda.show_topics(formatted=False)

    fig, axes = plt.subplots(2, 2, figsize=(10,10), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')


    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.savefig('./tmp/'+user_info['username']+'.jpg')
    image_rad = Image.open('./tmp/'+user_info['username']+'.jpg')
    user_info['image_rad'] = im_2_b64(image_rad)

    return user_info








