import eval_metrics as em
import numpy as np

fname = []

with open('./stanford_data/val.txt') as f:
	for line in f:
		fname.append(line[:-1])

s = 0
k = 0
for fn in fname:
	a = np.load('/home/shubham/iccv09Data/region-npy/{}_label.npy'.format(fn))
	b = np.load('/home/shubham/out-ep35-full-original/{}_pred.npy'.format(fn))
	b = np.load('/home/shubham/out-ep25-fe-batchnorm/{}_pred.npy'.format(fn))
	s += em.mean_IU(b,a)
	k += 1

print s
print k
print s/float(k)

