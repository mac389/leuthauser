import lda, csv

import numpy as np

from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

stopwords = set(open('stopwords.txt').read().splitlines())
data = filter(lambda x: x != 'none',[shift['Student Comment'].lower().strip() for shift in csv.DictReader(open('comments.csv'))])
data = [' '.join([word for word in string.split() if word not in stopwords]) for string in data]
tfx = TfidfVectorizer(data,tokenizer=word_tokenize,strip_accents='unicode',
	ngram_range=(1,3),min_df=3, use_idf=True)
tfidf = tfx.fit_transform(data)

model = lda.LDA(n_topics=3, n_iter=1000,random_state=1)
model.fit(tfidf)

'''
topic_word = model.topic_word_
n_top_words = 10

with open('lda-topics_short-quick-run.txt','wb') as outfile:
	for i,topic_dist in enumerate(topic_word):
		topic_words = np.array(data)[np.argsort(topic_dist)][:-n_top_words:-1]
		print>>outfile,'Topic {}: {}'.format(i, ' '.join(topic_words))
'''