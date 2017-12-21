import matplotlib.pyplot as plt

n_episodes = 300
mean_rewards = []
err = []

mean_rewards2 = []
err2 = []


base_dir = './final/'
b = 'CartPole'
#b = 'MountainCar'
filename = 'out_{}_{}.csv'.format('QLambda',b)
filename2 = 'out_{}_{}.csv'.format('QL Residual gradient',b)
# MountainCar
# CartPole
# Acrobot
# Gridworld

fname = base_dir + filename
fname2 = base_dir + filename2

fname = 'history_bn.log'

'epoch,acc,loss,sparse_categorical_accuracy,val_acc,val_loss,val_sparse_categorical_accuracy'
'0      1    2           3                     4        5      '

epochs = []
acc = []
train_loss = []
val_loss = []
val_acc = []

with open(fname,'rb') as f:

	for line in f:
		if line[0]=='e':
			continue
		a = line.split(',')
		a[1] = a[1][:-1]
		epochs.append(float(a[0]))
		acc.append(float(a[1]))
		train_loss.append(float(a[2]))

		val_acc.append(float(a[4]))
		val_loss.append(float(a[5]))
		print '--',a
		print len(a)


# with open(fname2,'rb') as f:

# 	for line in f:
# 		if line[0]=='M':
# 			continue
# 		a = line.split(',')
# 		a[1] = a[1][:-1]
# 		mean_rewards2.append(float(a[0]))
# 		err2.append(float(a[1]))
# 		print '--',a
# 		print len(a)


# print len(mean_rewards)
# print len(err)
# print mean_rewards[299]
# print err[299]
# print type(mean_rewards[299])
# print type(err[0])

plt.figure()
#plt.errorbar(range(200),mean_rewards,yerr=err , label='Q_learning')
#plt.errorbar(range(200),mean_rewards2,yerr=err2, label='Residual QL')
#plt.ylim(ymin=-10,ymax=800)
#plt.xlim(xmin=0)
plt.plot(epochs,acc,label='Training Accuracy')
plt.plot(epochs, val_acc,label='Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Number of Epochs')
plt.title('Accuracy for Front End Training only')#with Context Module')

plt.legend(loc='lower right')
plt.show()