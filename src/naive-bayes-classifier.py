import csv, random, nltk.classify

import utils as tech 

from awesome_print import ap 
from nltk import NaiveBayesClassifier
from nltk.util import ngrams
from nltk.metrics import scores

from nltk.classify import SklearnClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef

pipeline = Pipeline([('tfidf', TfidfTransformer()),
                     ('chi2', SelectKBest(chi2, k=1000)),
                     ('nb', MultinomialNB())])
classif = SklearnClassifier(pipeline)

def extract_featurelabel(student):
	features = {}
	bigrams = ngrams(student['Student Comment'],2)
	for word in student['Student Comment']:
		features['contains (%s)'%word] = word
	for bigram in bigrams:
		features['contains (%s)'%(' '.join(bigram))] = ' '.join(bigram)

	return (features,student['Physician Comment'])
	#return (features,find_student_grade(student['Name']))

def find_student_grade(name): 
	#Assumes that a dictionary with each student's name and grade has been created
	#Only look for last name because the first name was not recorded for all students
	#and looking for abbreviations does not help

	return [student['grade'] if student['name'] in name else False 
			for student in ratings][0]

def extract_grade(student):
	return {'name':student['name'].split(',')[0].capitalize(),
			'grade':student['grade']}

ratings = [extract_grade(student) for student in list(csv.DictReader(open('data.csv','rb')))]

data = list(csv.DictReader(open('comments.csv','rb')))
data = [student for student in data if student['Physician Comment']!='NA'
	and student['Physician Comment'] !='Cedar' and student['Physician Comment'] != 'X']

rating_names = [student['name'] for student in ratings]
data_names = list(set([student['Name'] for student in data]))
#cleans text for classifying
for i,student in enumerate(data):
	text = tech.cleanse(student['Student Comment'])
	data[i]['Student Comment'] = text

#split into testing and training sets
n = len(data)
test_idx = random.sample(xrange(n),int(n*0.5))
train_idx = set(xrange(n))-set(test_idx)

test_set = filter(lambda item: item[1] ,map(extract_featurelabel,[data[i] for i in test_idx]))
train_set = filter(lambda item: item[1] ,map(extract_featurelabel,[data[i] for i in train_idx]))

#classifier = NaiveBayesClassifier.train(train_set)
classif.train(test_set)
#Compute accuracy
test_data,test_label = zip(*test_set)
train_data,train_label = zip(*train_set)

predictions = classif.classify_many(test_data)

print confusion_matrix(test_label,predictions)
print matthews_corrcoef(test_label,predictions)
'''
#Only work if using built-in NLTK classifier
print ('Accuracy: {0:.2f}%'.format(100 * nltk.classify.accuracy(classif, test_set)))
classif.show_most_informative_features(20)
'''