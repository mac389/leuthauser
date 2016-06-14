import nltk, random, json

#NAIVE BAYES CLASSIFIER WORK?

def extract_features(list_of_words):
	return {'occurence_of_%s'%word:1 for word in list_of_words}

db = json.load(open('../data/comments.json','rb'))

random.shuffle(db)

testing_set = db[(len(db)/2):]
training_set = db[:(len(db)/2)]

training_set = [(extract_features(doc['cleansed text']),doc['Physician Comment']) for doc in training_set
			if doc['Physician Comment'] in ['Manager','Superior','Reporter']]
testing_set = [(extract_features(doc['cleansed text']),doc['Physician Comment']) for doc in testing_set
			if doc['Physician Comment'] in ['Manager','Superior','Reporter']]



classifier = nltk.NaiveBayesClassifier.train(training_set)
classifier.show_most_informative_features(5)
print nltk.classify.accuracy(classifier, testing_set)