# leuthauser

This project aims to identify whether the patterns of writing of students reflect the evaluations of attendings in a fourth year emergency medicine clerkship. Our underlying conceptual hypothesis is that students reflect their attitude to learning in their communication and that attendings evalaute medical students mostly on their attitude. (This assumes a minimal ammount of medical knowledge. Actual clinical knowledge was assessed on another test.)

We used Biterm Topic Modeling (Yan et al, 2013) to identify combinations of words that occur in the same context. 

The model calculates three matrices:

1.  *.pw_z : A matrix of _topics_ x _words_ where entry _ij_ indicates the posterior probability that word _j_ is associated with topic _i_. 
1. *.pz : A vector of _topics_ where entry _i_ indicates the prior probability of a word occurring.
1. *.pz_d : A matrix of _documents_ x _topics_ where entry _ij_ indicates the proportion to which topic _j_ contributes to document _i_. 



##Quickstart
     python setup.py
     

##Analysis
Comments are stored in `comments.csv`

##References
> Xiaohui Yan, Jiafeng Guo, Yanyan Lan, Xueqi Cheng. A Biterm Topic Model For Short Text. WWW2013.