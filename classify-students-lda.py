import csv, nltk, textmining, string

import numpy as np 

from awesome_print import ap
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
READ = 'rb'
stopwords = open('stopwords.txt').read().splitlines()

with open('comments.csv',READ) as csvfile:
	comments = [row for row in csv.DictReader(csvfile)]	

students =  {entry['Name']:"" for entry in comments}

def process(text):
	return ' '.join([nltk.stem.WordNetLemmatizer().lemmatize(word) for word in word_tokenize(text.decode('utf-8').encode('ascii','ignore').lower()) 
			if word not in stopwords and word not in ['x','na'] and word not in set(string.punctuation)])


for student in students:
	students[student] = process(' '.join([entry['Student Comment'] for entry in comments
								if entry['Name']==student]))

#ap(students)
keys = [student for student in students.iterkeys() if len(students[student]) > 0]
tfx = TfidfVectorizer([students[student] for student in keys],tokenizer=word_tokenize,strip_accents='unicode',
	min_df=3, use_idf=True)

tfidf = tfx.fit_transform([students[student] for student in keys])
lda = LatentDirichletAllocation(n_topics=8,max_iter=5,learning_method='online')
model = lda.fit_transform(tfidf)
'''
ap(dir(tfx))
ap(dict(zip(tfx.get_feature_names(),tfx.idf_)))
'''
#--lda
feature_names = tfx.get_feature_names()
feature_weights = tfx.idf_

n_top_words = 20
features = zip(feature_weights,feature_names)
with open('sklearn-lda-topics-w-weights','wb') as f:
	for topic_idx, topic in enumerate(lda.components_):
		print>>f, " : ".join(["%.4f * %s"%(features[i][0],features[i][1])
			for i in topic.argsort()[:-n_top_words - 1:-1]])