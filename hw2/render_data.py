import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

depth_losses = np.load('dnn_losses.npz')
trd = depth_losses["train"]
ted = depth_losses["test"]
epochs = np.arange(1, trd.shape[0]+1)

sns.lineplot(x = epochs, y = trd, label = "Train Loss")
sns.lineplot(x = epochs, y = ted, label = "Test Data")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.yscale("log")
plt.title("Training vs Test Loss")
plt.legend()
plt.show()

