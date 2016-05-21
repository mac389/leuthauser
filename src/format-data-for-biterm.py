import json

comments = json.load(open('../data/comments.json','rb'))
for_biterm = [comment['cleansed text'] for comment in comments]

with open('../../BTM/sample-data/for-biterm','wb') as fid:
	for document in for_biterm:
		print>>fid, ' '.join(document) if len(document) > 0 else '\n' 