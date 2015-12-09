import csv, json
import matplotlib.pyplot as plt 
import Graphics as artist
import numpy as np 

from awesome_print import ap 
from scipy.stats import linregress, ks_2samp
from matplotlib import rcParams

#rcParams['text.usetex'] = True
#plt.xkcd()
data = list(csv.DictReader(open('data.csv','rb')))
comments = list(csv.DictReader(open('comments.csv','rb')))

def pop(lst):
	if len(lst) > 0:
		ans = lst.pop()
		if ans != '':
			return ans
		else:
			return None
	else:
		return None


names = set(student['Name'] for student in comments)

#Condense all comments into a dictionary with student name at keys
condensed_comments = {name:{'comments':[student['Student Comment'] for student in comments if student['Name']==name]}
						for name in names}

for name in condensed_comments:
	comments = condensed_comments[name]['comments']
	condensed_comments[name]['completion'] = 1-(comments.count('None')+comments.count('NA')+comments.count(''))/float(len(comments))
	condensed_comments[name]['lengths'] = [len(comment.split()) for comment in comments]


unified_database = {name:(condensed_comments[name]['completion'], condensed_comments[name]['lengths'],
						pop([student['exam']  for student in data if name.lower().split()[-1] in student['name']])) 
						for name in names}


#Split based on completion score
upper_half = {key:value for key,value in unified_database.iteritems() if value[0]>0.5}
lower_half = {key:value for key,value in unified_database.iteritems() if value[0]<0.5}

print '-------Testing--------'
upper_performance = filter(None,[float(x[-1]) if x[-1] is not None else None for x in upper_half.itervalues()])
lower_performance = filter(None,[float(x[-1]) if x[-1] is not None else None for x in lower_half.itervalues()])
print ks_2samp(upper_performance,lower_performance)
print np.median(upper_performance), 0.5*np.subtract(*np.percentile(upper_performance, [75, 25]))
print np.median(lower_performance), 0.5*np.subtract(*np.percentile(lower_performance, [75, 25]))
print '-------Testing--------'
json.dump(upper_half,open('upper_half.json','wb'))
json.dump(lower_half,open('lower_half.json','wb'))



fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111)
bp = ax.boxplot([upper_performance,lower_performance],notch=True,patch_artist=True,
	positions=[0.5,1])
for box in bp['boxes']:
    # change outline color
    box.set( color='#7570b3', linewidth=2)
    # change fill color
    box.set( facecolor = '#1b9e77' )


## change color and linewidth of the whiskers
for whisker in bp['whiskers']:
    whisker.set(color='#7570b3', linewidth=2)

## change color and linewidth of the caps
for cap in bp['caps']:
    cap.set(color='#7570b3', linewidth=2)

## change color and linewidth of the medians
for median in bp['medians']:
    median.set(color='black', linewidth=2)

## change the style of fliers and their fill
for flier in bp['fliers']:
    flier.set(marker='o', color='#e7298a', alpha=0.5)
artist.adjust_spines(ax)
#ax.set_xticklabels(map(artist.format,['Completers','Non-completers']),rotation='vertical')
ax.set_xticklabels(['> 0.5' ,'< 0.5'])
ax.set_xlabel('Fraction of comments filled out')
ax.set_xlim(xmin=0.3,xmax=1.4)
ax.set_ylabel('Final Grade')
plt.tight_layout()
#plt.savefig('completers-noncompleter-exam-performance-xkcd.png')
plt.savefig('boxplot-no-xkcd.png')
del fig,ax
'''
for student,score in unified_database.items():
	if score[-1] == None:
		del unified_database[student]

completion,lengths,grades = zip(*unified_database.values())
grades = np.array(map(float,grades))
_lengths = np.array([np.average(lst) for lst in lengths])

completers,noncompleters = [],[]

for i in xrange(len(lengths)):
	completers.append((grades[i],np.median(lengths[i]))) if completion[i] > 0.5 else noncompleters.append((grades[i],np.median(lengths[i])))

fig = plt.figure(figsize=(5.5,6))
ax = fig.add_subplot(111)

completer_grade,completer_len = zip(*completers)
noncompleter_grade,noncompleter_len = zip(*noncompleters)


ax.scatter(completer_grade,completer_len,c='.9',s=30,label='>50%')
ax.scatter(noncompleter_grade,noncompleter_len,c='.1',s=30,label='<50%')
slope, intercept, r_value, p_value, std_err = linregress(grades,_lengths)
ax.plot(grades,slope*grades+intercept,'k--')

ax.annotate('$y=%.02f \cdot x %.02f$'%(slope,intercept), xy=(.45, .7),  xycoords='axes fraction',
            horizontalalignment='right', verticalalignment='top')

ax.annotate('$r^2=%.02f, p=%.02f$'%(np.corrcoef(grades,_lengths)[0,1],p_value), xy=(.45, .65),  xycoords='axes fraction',
        horizontalalignment='right', verticalalignment='top')

artist.adjust_spines(ax)
ax.set_xlabel('Final grade')
ax.set_ylabel('Median Length of Comment')

plt.legend(frameon=False)
plt.tight_layout()
plt.savefig('length-v-grade.png')
'''
'''
fig,axs = plt.subplots(nrows=1,ncols=2, sharex=True)
for ax,data,label in zip(axs,[completion,_lengths],['Fraction of completed comments','Avg. Length of Comment']):
	ax.scatter(grades,data,c=['.9' if x > 0.5 else '.1' for x in completion],s=30)

	slope, intercept, r_value, p_value, std_err = linregress(grades,data)
	ax.plot(grades,slope*grades+intercept,'k--')


	ax.annotate('$y=%.02f \cdot x %.02f$'%(slope,intercept), xy=(.55, .7),  xycoords='axes fraction',
            horizontalalignment='right', verticalalignment='top')

	ax.annotate('$r^2=%.02f, p=%.02f$'%(np.corrcoef(grades,data)[0,1],p_value), xy=(.55, .65),  xycoords='axes fraction',
            horizontalalignment='right', verticalalignment='top')
	

	artist.adjust_spines(ax)
	ax.set_xlabel('Final grade')
	ax.set_ylabel(label)

plt.tight_layout()
#plt.show()
plt.savefig('correlation-completion-ratings.tiff')
'''
