def weighted_jaccard_similarity(fdist1,fdist2, word_count=20): #more elegant would be to use None

    '''
        The weighted jaccard similarity, aka similarity ratio, 

    '''
    freq1 = dict(fdist1.most_common(word_count))
    freq2 = dict(dist2.most_common(word_count))


    #find words common to both 
    common_words = set(words1) & set(words2)

    #find frequencies of words common to both 