from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer
import pandas as pd
import re
import numpy as np
import warnings
import matplotlib as plt
warnings.filterwarnings('ignore')


csv = pd.read_csv("train.csv")
train_set = csv['Comment'].values[0]
comments = []
for i in range(10): #len(train_set)
    comments.append(csv['Comment'].values[i])


comments2 = []
wordSet = []
for i in range(10):
    comments2.append(re.sub(r"[^a-zA-Z0-9]", " ", csv['Comment'].values[i].lower()).split())
    wordSet = np.union1d(wordSet, comments2[i])

# DF:
# vectorizer = CountVectorizer(stop_words='english')
# X = vectorizer.fit_transform(comments)
# df_bow_sklearn = pd.DataFrame(X.toarray(),columns=vectorizer.get_feature_names_out())
# df_bow_sklearn.head()
# print(df_bow_sklearn.head())

#TF-IDF:
tr_idf_model  = TfidfVectorizer(stop_words='english')
tf_idf_vector = tr_idf_model.fit_transform(comments)
tf_idf_array = tf_idf_vector.toarray()
words_set = tr_idf_model.get_feature_names_out()
df_tf_idf = pd.DataFrame(tf_idf_array, columns = words_set)

w2v = Word2Vec(comments2,window=5,min_count=1)
words = list(w2v.wv.index_to_key)



# for word in words:
#     X = w2v.wv.get_index(word)
# pca = PCA(n_components=2)
# result = pca.fit_transform(X)
# # create a scatter plot of the projection
# plt.scatter(result[:, 0], result[:, 1])
# words = list(w2v.wv.index_to_key)
# for i, word in enumerate(words):
#    plt.annotate(word, xy=(result[i, 0], result[i, 1]))
# plt.show()










# import pandas as pd
# import numpy as np
# import re
#
# def calculateBOW(wordset,l_doc):
#   tf_diz = dict.fromkeys(wordset,0)
#   for word in l_doc:
#       tf_diz[word]=l_doc.count(word)
#   return tf_diz
#
# comments = []
# wordSet =[]
# bows = []
#
# csv = pd.read_csv("train.csv")
#
# x_train = csv[['Id','Comment','Topic']]
# train_set = csv['Comment'].values[0]
#
# for i in range(len(train_set)):
#     comments.append(re.sub(r"[^a-zA-Z0-9]", " ", csv['Comment'].values[i].lower()).split())
#     wordSet = np.union1d(wordSet, comments[i])
#
# for i in range(len(train_set)):
#     bows.append(calculateBOW(wordSet, comments[i]))
#
# df_bow = pd.DataFrame(bows)
# df_bow.head()
#
# print(df_bow)