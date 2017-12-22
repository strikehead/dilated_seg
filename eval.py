import eval_metrics as em
import numpy as np

fname = []

with open('./iccv09Data/val.txt') as f:
	for line in f:
		fname.append(line[:-1])

s = 0
k = 0
basedir = '/home/shubham/'
for fn in fname:
	a = np.load(basedir + '/iccv09Data/region-npy/{}_label.npy'.format(fn))
	b = np.load(basedir + '/out-ep35-full-original/{}_pred.npy'.format(fn))
	b = np.load(basedir + '/out-ep25-fe-batchnorm/{}_pred.npy'.format(fn))
	s += em.mean_IU(b,a)
	k += 1


print "Average IoU : ",s/float(k)

