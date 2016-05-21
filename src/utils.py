import string, nltk, itertools
import numpy as np 

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from scipy.stats import percentileofscore
from awesome_print import ap 

stopwords = set(open('../data/stopwords.txt').read().splitlines())
punkt = set(string.punctuation)
lemmatizer = WordNetLemmatizer()

def jaccard_similarity(one,two):
        one = set(one)
        two = set(two)
        return len(set(one) & set(two))/float(len(set(one) | set(two)))

def calculate_jmat(samples):
    return[jaccard_similarity(one,two) for one, two in itertools.combinations(samples.values(),2)]


def calculate_pvalue(dist,score):
    p = percentileofscore(dist,score)
    return 0.01*min(p,100-p)

def cleanse(text):
    text = ''.join(ch for ch in text.lower() if not any([ch in punkt,ch.isdigit(),ch == 'none']))
    text = [lemmatizer.lemmatize(word,pos=get_wordnet_pos(pos)) 
            for word,pos in nltk.pos_tag([word for word in nltk.word_tokenize(''.join(ch for ch in text if ord(ch)<128))
            if word not in stopwords and word != 'na'])]
    return text

def weighted_jaccard_similarity(fdist1,fdist2, word_count=20): #more elegant would be to use None

    '''
        The weighted jaccard similarity, aka similarity ratio, 

    '''
    freq1 = dict(fdist1.most_common(word_count))
    freq2 = dict(fdist2.most_common(word_count))

    #Don't forget normalization factor
    normalizations = {'freq1':sum(freq1.values()), 'freq2':sum(freq2.values())}
    #find words common to both 
    common_words = set(freq1.keys()) & set(freq2.keys())
    common = sum([freq1[word]*freq2[word] for word in common_words])

    in_a = np.square([freq1.values()]).sum()
    in_b = np.square([freq2.values()]).sum()

    return float(common)/(in_a + in_b - common)

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def savelines(lst,filename='out'):
    with open(filename,'wb') as f:
        for line in lst:
            print>>f,line

def save_ngrams(lst,filename='out'):
    with open(filename,'wb') as f:
        for word,freq in lst:
            print>>f,'%s (%d)'%(word,freq)