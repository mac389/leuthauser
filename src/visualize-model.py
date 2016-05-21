import numpy as np 
import matplotlib.pyplot as plt 
import Graphics as artist

from matplotlib import rcParams

rcParams['text.usetex'] = True

data = np.loadtxt('../data/model/model/k20.pw_z')
data[data>0.01] = 0

#--- Visualize words
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.imshow(data,interpolation='nearest',aspect='auto',cmap=plt.cm.bone_r)
artist.adjust_spines(ax)
ax.set_ylabel(artist.format('Topics'))
ax.set_xlabel(artist.format('Words'))
ax.set_xticks([])
ax.set_yticks([])
plt.colorbar(cax)
plt.tight_layout()
plt.show()

#--Visualize words
data = np.loadtxt('../data/model/model/k20.pz_d')
data[data>0.01] = 0
print data.shape

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.imshow(data,interpolation='nearest',aspect='auto',cmap=plt.cm.bone_r)
artist.adjust_spines(ax)
ax.set_xlabel(artist.format('Topics'))
ax.set_ylabel(artist.format('Comments'))
ax.set_xticks([])
ax.set_yticks([])
plt.colorbar(cax)
plt.tight_layout()
plt.show()
