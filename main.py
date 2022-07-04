import math
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re
import numpy as np
import warnings
from nltk.corpus import stopwords

warnings.filterwarnings('ignore')


csv = pd.read_csv("train.csv")
train_set = csv['Comment']

comments = []
for i in range(len(train_set)):
    comments.append(csv['Comment'].values[i])

comments2 = []
wordSet = []
en_stops = set(stopwords.words('english'))

for i in range(len(train_set)):
    temStr = re.sub(r"[^a-zA-Z0-9]", " ", csv['Comment'].values[i].lower()).split()
    hasStopWord = True
    while(hasStopWord):
        hasStopWord = False
        for word in temStr:
            if word in en_stops or len(word) == 1 or len(word) == 2:
                temStr.remove(word)
                hasStopWord = True
    comments2.append(temStr)
    wordSet = np.union1d(wordSet, comments2[i])

minLen = 1000
indexI = 0
indexJ = 0

for i in range(len(comments2)):
    for j in range(len(comments2[i])):
        if comments2[i][j] == 'n':
            print(comments2[i][j])
        if len(comments2[i][j]) < minLen:
            minLen = len(comments2[i][j])
            indexI = i
            indexJ = j

# # DF:
# vectorizer = CountVectorizer(stop_words='english')
# X = vectorizer.fit_transform(comments)
# df_bow_sklearn = pd.DataFrame(X.toarray(),columns=vectorizer.get_feature_names_out())
# df_bow_sklearn.head()
# #print(df_bow_sklearn.head())
#

#TF-IDF:
tr_idf_model  = TfidfVectorizer(stop_words='english')
tf_idf_vector = tr_idf_model.fit_transform(comments)
tf_idf_array = tf_idf_vector.toarray()
words_set = tr_idf_model.get_feature_names_out()
df_tf_idf = pd.DataFrame(tf_idf_array, columns = words_set)


w2v = Word2Vec(comments2,window=7,min_count=1)
vector = w2v.wv['good']  # get numpy vector of a word
sims = w2v.wv.most_similar('good', topn=10)  # get other similar words
words = list(w2v.wv.index_to_key)




CTarget = 3
dictLabels = {}
N = 0
for i in range(5) : #range(len(comments2)):
    for j in range(len(comments2[i])):
        if comments2[i][j] in dictLabels.keys():
            continue
        dictLabels[comments2[i][j]] = [N,N,i,j]
        N += 1


k = 5
print('finished')
def kNearestN (k, i):
    d = {}
    for m in dictLabels:
        if m is not i:
            d[m] = w2v.wv.distance(i, m)
    d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1])}
    return list(d.items())[0:k]

g = 1.5

CPrevious = N
CCurrent = math.floor(N/g)
Dcurrent = [[0] * N] * N
sum = 0
for i in dictLabels:
    for j in dictLabels:
        x = w2v.wv.most_similar(i, topn=k)
        x.append((i,0))
        y = w2v.wv.most_similar(j, topn=k)
        y.append((j,0))
        for n in range(len(x)):
            for m in range(len(y)):
                sum += w2v.wv.distance(x[n][0], y[m][0])
        Dcurrent[dictLabels[i][0]][dictLabels[j][0]] = (1.0 / math.pow((k + 1), 2)) * sum


print("im hereeee")
def maxMin (K, S, distances):
    tmpMax = {}
    tmpMin = {}

    for j in S:
        for k in K:
            tmpMin[k]=distances[k][j]
        tmpMin = {k: v for k, v in sorted(tmpMin.items(), key=lambda item: item[1])}
        tmpMax[list(tmpMin)[0]] = tmpMin.get(list(tmpMin)[0])

    tmpMax = {k: v for k, v in sorted(tmpMin.items(), key=lambda item: item[1],reverse=True)}
    return list(tmpMax)[0]

def identificationOfKeyClusters(c,distances):
    m = len(distances)
    I = {}
    for i in range(m):
        sum = 0
        for j in range(m):
            sum += distances[i][j]
        I[i] = sum / m
    I = {k: v for k, v in sorted(I.items(), key=lambda item: item[1])}

    S = []
    S.append(list(I)[0])

    K = [i for i in range(0, m) if i not in S]
    n = 1

    while n != c:
        tmp = maxMin(K, S, distances)
        S.append(tmp)
        K = [i for i in range(0, m) if i not in S]
        n += 1
    return S


print('im here')
while CCurrent > CTarget:
    SCurrent = identificationOfKeyClusters(CCurrent,Dcurrent)
    savedLable = SCurrent[0]
    distance = 100000
    for n in dictLabels:
        if n is not SCurrent:
            for m in SCurrent:
                if Dcurrent[dictLabels[n][1]][m] < distance:
                    distance = Dcurrent[dictLabels[n][1]][m]
                    savedLable = m
                    print(savedLable)
            dictLabels[n][1] = SCurrent[savedLable]

    for i in dictLabels:
        for j in dictLabels:
            pi = []
            piSet = set()
            pj = []
            pjSet = set()
            for m in dictLabels:
                if dictLabels[m][1] == dictLabels[i][1]:
                    pi.append(m)
                    piSet.add(m)
            for m in range(len(pi)):
                W = w2v.wv.most_similar(pi[m], topn=k)
                for w in W:
                    piSet.add(w[0])
            for m in dictLabels:
                if dictLabels[m][1] == dictLabels[j][1]:
                    pj.append(m)
                    pjSet.add(m)
            for m in range(len(pj)):
                W = w2v.wv.most_similar(pj[m], topn=k)
                for w in W:
                    pjSet.add(w[0])
            sum = 0
            for a in piSet:
                for b in pjSet:
                    sum += w2v.wv.distance(a,b)
            Dcurrent[dictLabels[i][0]][dictLabels[j][0]] = (1.0 / (len(piSet) * len(pjSet))) * sum
    CPrevious = CCurrent
    CCurrent = math.floor(CCurrent/g)

Sfinal = identificationOfKeyClusters(CTarget,Dcurrent)
savedLable = Sfinal[0]
distance = 1000
for n in dictLabels:
    if n is not Sfinal:
        for m in Sfinal:
            if Dcurrent[dictLabels[n][1]][m] < distance:
                distance = Dcurrent[dictLabels[n][1]][m]
                savedLable = m
        dictLabels[n][1] = Sfinal[savedLable]

for n in dictLabels:
    i = 0
    print(n + " lable is:" + dictLabels[n][1])
    i += 1

label1 = -1
label2 = -1
label3 = -1
for m in dictLabels:
    if label1 == -1:
        label1 = dictLabels[m][1]
    if dictLabels[m][1] != label1 and label2 == -1:
        label2 = dictLabels[m][1]
    if dictLabels[m][1] != label1 and dictLabels[m][1] != label2:
        label3 = dictLabels[m][1]
        break

label1Weight = 0.0
label2Weight = 0.0
label3Weight = 0.0
finalLabels = [0] * N
for m in range(20): #range(len(comments2))
    for n in range(len(comments2[m])):
        if dictLabels[comments2[m][n]] == label1:
            label1Weight += df_tf_idf.get(comments2[m][n])[m]
        elif dictLabels[comments2[m][n]] == label2:
            label2Weight += df_tf_idf.get(comments2[m][n])[m]
        else:
            label3Weight += df_tf_idf.get(comments2[m][n])[m]
    if label3Weight > label1Weight and label3Weight > label2Weight:
        finalLabels[m] = label3    # 2
    elif label2Weight > label1Weight and label2Weight > label3Weight:
        finalLabels[m] = label2    # 1
    else:
        finalLabels[m] = label1    # 0
    label1Weight = 0.0
    label2Weight = 0.0
    label3Weight = 0.0

for m in range(len(comments2)):
    print(m," label:", finalLabels[m])


topics = []
listLabel = [[0, 0, 0]] * 3
maxB =0
maxP =0
maxC =0
maxList = [0, 0, 0]

for i in range(comments2):
    topic = csv['Topic'].values[i]
    topics.append(csv['Topic'].values[i])

    if topic == "Biology":
        listLabel[0][finalLabels[i]] += 1
        maxB = max(listLabel[0])

    if topic == "Physics":
        listLabel[1][finalLabels[i]] += 1
        maxP = max(listLabel[1])

    if topic == "Chemistry":
        listLabel[2][finalLabels[i]] += 1
        maxC = max(listLabel[2])

    maxList = [maxB,maxP,maxC]

maxV = max(maxList)
predictLabel = [0, 0, 0]

maxV = max(maxList)
for x in range(3): #baraye peida kardan 3 label
    for i in range(3):
        if listLabel[i] == [0, 0, 0]:
            continue
        for j in range(3):
            if maxV == listLabel[i][j]:
                predictLabel[i] = j
                listLabel[i] = [0, 0, 0]
    maxList = []
    for i in range(3):
        if listLabel[i] == [0, 0, 0]:
            continue
        maxList.append(max(listLabel[i]))
    maxV = max(maxList)



csv = pd.read_csv("test.csv")
topics = csv['Topic']

correct = 0
for topic in topics:

    if finalLabels[i] == predictLabel[0]:
        predicTopic = "Biology"

    elif finalLabels[i] == predictLabel[1]:
        predicTopic = "Physics"

    elif finalLabels[i] == predictLabel[2]:
        predicTopic = "Chemistry"

    if topic == predicTopic:
        correct += 1

accuracy = correct / len(topics)
print(accuracy)

