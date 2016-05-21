import csv, json 

#Load comments
with open('../data/comments.csv','rb') as fid:
	reader = csv.DictReader(fid)
	comments = [row for row in reader]

json.dump(comments,open('../data/comments.json','wb'))