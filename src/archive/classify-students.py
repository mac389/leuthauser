import csv, lda, nltk 

from awesome_print import ap
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

READ = 'rb'
stopwords = open('stopwords.txt').read().splitlines()

with open('comments.csv',READ) as csvfile:
	comments = [row for row in csv.DictReader(csvfile)]	

students =  {entry['Name']:"" for entry in comments}

def process(text):
	
	return ' '.join([nltk.stem.WordNetLemmatizer().lemmatize(word) for word in word_tokenize(text.decode('utf-8').encode('ascii','ignore').lower()) 
			if word not in stopwords and word not in ['x','na']])


for student in students:
	students[student] = process(' '.join([entry['Student Comment'] for entry in comments
								if entry['Name']==student]))

ap(students)
keys = [student for student in students.iterkeys() if len(students[student]) > 0]
tfx = TfidfVectorizer([students[student] for student in keys],tokenizer=word_tokenize,strip_accents='unicode',
	min_df=3, use_idf=True)

tfidf = tfx.fit_transform([students[student] for student in keys])

#--LSA
pca = PCA(n_components=3)
model = pca.fit_transform(tfidf.toarray())
ap(pca.explained_variance_ratio_)

#--lda
model = lda.LDA(n_topics=3, n_iter=1000,random_state=1)
model.fit(tfidf)

topic_word = model.topic_word_
for i,topic_dist in enumerate(topic_word):
	words = np.array()