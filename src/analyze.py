import csv, skll

import numpy as np
import matplotlib.pyplot as plt
import rpy2.robjects as robjects

from rpy2.robjects.numpy2ri import numpy2ri
from rpy2.robjects.packages import importr
from scipy.stats import fisher_exact

valid_fields = ['name','amy','kevin','exam','grade']
data =[{field:row[field] for field in valid_fields} for row in csv.DictReader(open('data.csv'))]
filtered_data  = [entry for entry in data if not any([field=='' for field in entry.values()])]

with open('output-external-cutoffs','w') as f:

	print>>f,'---------'
	print>>f, 'Total records: %d'%len(data)
	print>>f, 'Complete records: %d'%len(filtered_data)
	print>>f, '---------'


	contingency_table = np.array([[len([datum for datum in data if datum['kevin']==str(i) and datum['amy']==str(j)]) 
						for i in range(4)] for j in range(4)])

	filtered_contingency_table = np.array([[len([datum for datum in filtered_data if datum['kevin']==str(i) and datum['amy']==str(j)]) 
						for i in range(4)] for j in range(4)])

	print>>f, 'With all data: '
	print>>f,(contingency_table)

	print>>f, 'With only full fields: '
	print>>f,(filtered_contingency_table)

	fisher_test = robjects.r['fisher.test']
	print>>f, fisher_test(numpy2ri(contingency_table),numpy2ri(filtered_contingency_table),simulate_p_value=True,B=6000)

	kevin,amy = zip(*[(int(datum['kevin']),int(datum['amy'])) for datum in data
						if datum['kevin']!='' and datum['amy']!=''])

	filtered_kevin,filtered_amy = zip(*[(int(datum['kevin']),int(datum['amy'])) for datum in filtered_data
						if datum['kevin']!='' and datum['amy']!=''])

	ratings = map(str,range(4))
	kevin_amy_contingency = np.array([[len([item for item in filtered_data if 
								 item['kevin'] == kevin_rating and item['amy'] == amy_rating]) 
								for kevin_rating in ratings] for amy_rating in ratings])

	print>>f, 'Did Kevin and Amy rate differently? %s'%fisher_test(numpy2ri(kevin_amy_contingency),simulate_p_value=True,B=6000)


	print>>f, '---For all data------'
	print>>f, 'Chance agreement: %.02f'%(np.trace(contingency_table)/float(contingency_table.sum()))
	print>>f,  "Cohen's kappa: %.02f"%skll.kappa(kevin,amy)
	print>>f, '---------'


	print>>f, '---For full fields-----'
	print>>f, 'Chance agreement: %.02f'%(np.trace(filtered_contingency_table)/float(filtered_contingency_table.sum()))
	print>>f,  "Cohen's kappa: %.02f"%skll.kappa(kevin,amy)
	print>>f, '---------'

	#--Does reflection related to grade?

	unfiltered_distribution_grades = [float(item['exam']) for item in data if item['exam'] != '']
	filtered_distribution_grades = [float(item['exam']) for item in filtered_data if item['exam'] != '']

	quartiles = [(0,70),(70,80),(80,90),(90,100)]
	unfiltered_quartiles = [(np.percentile(unfiltered_distribution_grades,lower),np.percentile(unfiltered_distribution_grades,upper))
								for (lower,upper) in quartiles]
	filtered_quartiles = [(np.percentile(filtered_distribution_grades,lower),np.percentile(filtered_distribution_grades,upper))
								for (lower,upper) in quartiles]

	def which_quartile(item,reference=quartiles):
		return [float(item) >=float(lower) and float(item )<= float(upper) for lower,upper in reference].index(True) #Assuming will only fall into one quartile

	def quartile(data,reference=quartiles):
		print [float(item) >=float(lower) and float(item )<= float(upper) for lower,upper in reference].index(True)
		return [which_quartile(float(item['exam']),quartiles) for item in filtered_data]

	reflection_quartiles = map(str,range(0,4))
	exam_quartiles = range(0,4)
	grades_reflection_contingency_tables = {rater:np.array([[len([item for i,item in enumerate(filtered_data) 
				if item[rater] == str(reflection_quartile) and which_quartile(item['exam']) == exam_quartile]) 
				for exam_quartile in exam_quartiles] for reflection_quartile in reflection_quartiles]) for rater in ['kevin','amy']}

	print>>f,(grades_reflection_contingency_tables)

	for rater,table in grades_reflection_contingency_tables.iteritems():
		print>>f, 'Rater %s: %s'%(rater,fisher_test(numpy2ri(table),simulate_p_value=True,B=6000))
