import os
import pandas as pd
import matplotlib.pyplot as plt

log_dir ='.\\log\\'
current = os.getcwd()

res_18_log = pd.read_csv(log_dir + 'ResNet18_3EPOCH_train_log.csv', index_col=0, header=None)
res_34_log = pd.read_csv(log_dir + 'ResNet34_3EPOCH_train_log.csv', index_col=0, header=None)
res_50_log = pd.read_csv(log_dir + 'ResNet50_3EPOCH_train_log.csv', index_col=0, header=None)
res_101_log = pd.read_csv(log_dir + 'ResNet101_3EPOCH_train_log.csv', index_col=0, header=None)
res_152_log = pd.read_csv(log_dir + 'ResNet152_3EPOCH_train_log.csv', index_col=0, header=None)

######################################
# Visualiziing training loss and acc #
######################################
""" If you want to compare another model's train acc/loss, activate below comments. """

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))


ax1.plot(res_18_log.iloc[:,0], linewidth=1, label='ResNet18')
ax1.plot(res_34_log.iloc[:,0], linewidth=1, label='ResNet34')
ax1.plot(res_50_log.iloc[:,0], linewidth=1, label='ResNet50')
ax1.plot(res_101_log.iloc[:,0], linewidth=1, label='ResNet101')
ax1.plot(res_152_log.iloc[:,0], linewidth=1, label='ResNet152')

ax1.set_title('Training Loss Graph', fontsize=15)
ax1.set_xlabel('Iteration', fontsize=15)
ax1.set_ylabel('Loss', fontsize=15)

fig.legend(fontsize=15)

ax2.plot(res_18_log.iloc[:,1], linewidth=1, label='ResNet18')
ax2.plot(res_34_log.iloc[:,1], linewidth=1, label='ResNet34')
ax2.plot(res_50_log.iloc[:,1], linewidth=1, label='ResNet50')
ax2.plot(res_101_log.iloc[:,1], linewidth=1, label='ResNet101')
ax2.plot(res_152_log.iloc[:,1], linewidth=1, label='ResNet152')

ax2.set_title('Training Accuracy Graph', fontsize=15)
ax2.set_xlabel('Iteration', fontsize=15)
ax2.set_ylabel('Accuracy', fontsize=15)

plt.show()


########################################
# Visualiziing validation loss and acc #
########################################
""" If you want to compare another model's validation acc/loss, activate below comments. """

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,8))

ax1.plot(res_18_log.iloc[:,2], linewidth=1, label='ResNet18')
ax1.plot(res_34_log.iloc[:,2], linewidth=1, label='ResNet34')
ax1.plot(res_50_log.iloc[:,2], linewidth=1, label='ResNet50')
ax1.plot(res_101_log.iloc[:,2], linewidth=1, label='ResNet101')
ax1.plot(res_152_log.iloc[:,2], linewidth=1, label='ResNet152')

ax1.set_title('Validation Loss Graph', fontsize=15)
ax1.set_xlabel('Iteration', fontsize=15)
ax1.set_ylabel('Loss', fontsize=15)

fig.legend(fontsize=15)

ax2.plot(res_18_log.iloc[:,3], linewidth=1, label='ResNet18')
ax2.plot(res_34_log.iloc[:,3], linewidth=1, label='ResNet34')
ax2.plot(res_50_log.iloc[:,3], linewidth=1, label='ResNet50')
ax2.plot(res_101_log.iloc[:,3], linewidth=1, label='ResNet101')
ax2.plot(res_152_log.iloc[:,3], linewidth=1, label='ResNet152')

ax2.set_title('Validation Accuracy Graph', fontsize=15)
ax2.set_xlabel('Iteration', fontsize=15)
ax2.set_ylabel('Accuracy', fontsize=15)

plt.show()