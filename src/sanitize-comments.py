import json
import utils as tech

comments = json.load(open('../data/comments.json','rb'))

for comment in comments:
	txt = comment['Student Comment']
	comment['cleansed text'] = tech.cleanse(txt)

json.dump(comments,open('../data/comments.json','wb'))