import json 
import seaborn as sns 
import matplotlib.pyplot as plt 

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA, LatentDirichletAllocation

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

db = json.load(open('../data/comments.json','rb'))
idx = [i for i in xrange(len(db)) if db[i]['Student Comment']!= 'NA' and db[i]['Student Comment']!='None']

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform([' '.join(db[i]['cleansed text']) for i in idx])

with open('../data/valid_idx','wb') as f:
	for item in idx:
		print>>f,idx
n_samples = 2000
n_features = 1000
n_topics = 10
n_top_words = 20
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=n_features,
                                stop_words='english')
tf = tf_vectorizer.fit_transform([' '.join(db[i]['cleansed text']) for i in idx])
lda = LatentDirichletAllocation(n_topics=10, max_iter=5,
                                learning_method='online', learning_offset=50.,
                                random_state=0)
XX = lda.fit_transform(tf)
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, 10)

pca = PCA(n_components=5)
YY = pca.fit_transform(XX)
print pca.explained_variance_ratio_

colors = {'Interpreter':'r', "Manager":'k',"Superior":'g','None':'w','Reporter':'w','Inadequate':'w',
	"Manager-Superior":'k','Interpreter-Manager':'r','Cedar':'w','Reporter-Interpreter':'w','NA':'w','X':'w'}

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(YY[:,2],YY[:,3],c=[colors[db[i]['Physician Comment']] for i in idx])
plt.show()
'''
PCA on TF_IDF failed

X = cosine_similarity(tfidf_matrix,tfidf_matrix)

print dict(zip(tfidf_vectorizer.get_feature_names(), tfidf_vectorizer.idf_))
pca = PCA(n_components=5)
Y = pca.fit_transform(tfidf_matrix.toarray())

print pca.explained_variance_ratio_
'''
'''
g = sns.clustermap(X)
g.savefig("../imgs/cluster-by-cosine.png")
'''