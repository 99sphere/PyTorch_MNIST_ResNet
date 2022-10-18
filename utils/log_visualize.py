import os
import pandas as pd
import matplotlib.pyplot as plt

log_dir = ".\\log\\"
current = os.getcwd()

my_res_18_log = pd.read_csv(
    log_dir + "ResNet18_train_log.csv", index_col=0, header=None
)
my_res_34_log = pd.read_csv(
    log_dir + "ResNet34_train_log.csv", index_col=0, header=None
)
my_res_50_log = pd.read_csv(
    log_dir + "ResNet50_train_log.csv", index_col=0, header=None
)
my_res_101_log = pd.read_csv(
    log_dir + "ResNet101_train_log.csv", index_col=0, header=None
)
my_res_152_log = pd.read_csv(
    log_dir + "ResNet152_train_log.csv", index_col=0, header=None
)

image_path = ".\\images\\"
summary = "All Models"

######################################
# Visualiziing training loss and acc #
######################################
""" If you want to compare another model's train acc/loss, activate below comments. """

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

ax1.plot(my_res_18_log.iloc[:, 0], linewidth=1, label="My ResNet18")
ax1.plot(my_res_34_log.iloc[:, 0], linewidth=1, label="My ResNet34")
ax1.plot(my_res_50_log.iloc[:, 0], linewidth=1, label="My ResNet50")
ax1.plot(my_res_101_log.iloc[:, 0], linewidth=1, label="My ResNet101")
ax1.plot(my_res_152_log.iloc[:, 0], linewidth=1, label="My ResNet152")

ax1.set_title("Training Loss Graph", fontsize=15)
ax1.set_xlabel("Iteration", fontsize=15)
ax1.set_ylabel("Loss", fontsize=15)

fig.legend(fontsize=15)

ax2.plot(my_res_18_log.iloc[:, 1], linewidth=1, label="My ResNet18")
ax2.plot(my_res_34_log.iloc[:, 1], linewidth=1, label="My ResNet34")
ax2.plot(my_res_50_log.iloc[:, 1], linewidth=1, label="My ResNet50")
ax2.plot(my_res_101_log.iloc[:, 1], linewidth=1, label="My ResNet101")
ax2.plot(my_res_152_log.iloc[:, 1], linewidth=1, label="My ResNet152")

ax2.set_title("Training Accuracy Graph", fontsize=15)
ax2.set_xlabel("Iteration", fontsize=15)
ax2.set_ylabel("Accuracy", fontsize=15)
plt.savefig(image_path + summary + "_Training_Accuracy_Graph")
plt.show()


########################################
# Visualiziing validation loss and acc #
########################################
""" If you want to compare another model's validation acc/loss, activate below comments. """

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

ax1.plot(my_res_18_log.iloc[:, 2], linewidth=1, label="My ResNet18")
ax1.plot(my_res_34_log.iloc[:, 2], linewidth=1, label="My ResNet34")
ax1.plot(my_res_50_log.iloc[:, 2], linewidth=1, label="My ResNet50")
ax1.plot(my_res_101_log.iloc[:, 2], linewidth=1, label="My ResNet101")
ax1.plot(my_res_152_log.iloc[:, 2], linewidth=1, label="My ResNet152")

ax1.set_title("Validation Loss Graph", fontsize=15)
ax1.set_xlabel("Iteration", fontsize=15)
ax1.set_ylabel("Loss", fontsize=15)

fig.legend(fontsize=15)

ax2.plot(my_res_18_log.iloc[:, 3], linewidth=1, label="My ResNet18")
ax2.plot(my_res_34_log.iloc[:, 3], linewidth=1, label="My ResNet34")
ax2.plot(my_res_50_log.iloc[:, 3], linewidth=1, label="My ResNet50")
ax2.plot(my_res_101_log.iloc[:, 3], linewidth=1, label="My ResNet101")
ax2.plot(my_res_152_log.iloc[:, 3], linewidth=1, label="My ResNet152")

ax2.set_title("Validation Accuracy Graph", fontsize=15)
ax2.set_xlabel("Iteration", fontsize=15)
ax2.set_ylabel("Accuracy", fontsize=15)

plt.savefig(image_path + summary + "_Validation_Accuracy_Graph")
plt.show()
