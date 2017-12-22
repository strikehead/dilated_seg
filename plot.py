import matplotlib.pyplot as plt


fname = 'history_bn.log'
'''
'epoch,acc,loss,sparse_categorical_accuracy,val_acc,val_loss,val_sparse_categorical_accuracy'
'0      1    2           3                     4        5      '
'''

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


plt.figure()
#plt.ylim(ymin=-10,ymax=800)
#plt.xlim(xmin=0)
plt.plot(epochs,acc,label='Training Accuracy')
plt.plot(epochs, val_acc,label='Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Number of Epochs')
plt.title('Accuracy for Front End Training only') #with Context Module')

plt.legend(loc='lower right')
plt.show()