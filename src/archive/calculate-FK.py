import csv
import numpy as np
import Graphics as artist
import matplotlib.pyplot as plt 

from matplotlib import rcParams
from textstat.textstat import textstat 

rcParams['text.usetex'] = True

data = [line for line in csv.DictReader(open('comments.csv','rb'))]
names = set([entry['Name'] for entry in data])
TAB = '\t'
WRITE = 'wb'
READ = 'rb'
filename = 'FK-calculated'

with open(filename,WRITE) as out:
	for name in names: 
		#This measurement is confounded by lengths of the text
		text = ' '.join([entry['Student Comment'] for entry in data if entry['Name'] == name and entry['Student Comment'] != 'None'])
		try:
			grade_level =  textstat.flesch_kincaid_grade(text)
		except: 
			grade_level = -1
		try:
			lex_div =  len(text.split())/float(len(set(text.split())))
		except:
			lex_div = -1 
		print>>out,'%s \t %.02f \t %.02f'%(name,grade_level,lex_div)


names, grade_levels, lex_div= zip(*[line.split('\t') for line in open(filename,READ).read().splitlines()])
grade_levels = map(float,grade_levels)
lex_div = map(float,lex_div)

fig,(ax,ax2) = plt.subplots(nrows=1,ncols=2,sharey=True)
ax.hist(grade_levels, bins=10,color='k')
artist.adjust_spines(ax)
ax.set_xlabel(artist.format('Flesch Kincaid Grade Level'))
ax.set_ylabel(artist.format('No. of students'))

ax2.hist(lex_div, bins=10,color='k')
artist.adjust_spines(ax2)
ax2.set_xlabel(artist.format('Lexical Diversity'))
ax2.set_ylabel(artist.format('No. of students'))
plt.tight_layout()
plt.savefig('ld-fk.png',dpi=300)