# FINAL CODE OUTPUTS
# 200 1-200 98.0 0.02
# 300 401-700 98.67 0.034
# 400 901-1300 95.0 0.05
# 500 501-1000 96.0 0.04
# 600 201-800 98.0 0.02
# 700 601-1300 94.86 0.051
# 800 401-1200 96.5 0.035
# 900 1-900 94.67 0.053
# 1000 338-1338 94.8 0.052
# Total 1-1338 93.73 0.062
#
# ORIGINAL CODE OUTPUTS
# 200 1-200 84.5 0.155
# 300 401-700 89.0 0.11
# 400 901-1300 85.5 0.145
# 500 501-1000 89.6 0.104
# 600 201-800 89.67 0.103
# 700 601-1300 87.86 0.121
# 800 401-1200 89.75 0.1025
# 900 1-900 89.78 0.1022
# 1000 338-1338 89.5 0.105
# Total 1-1338 88.56 0.114



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

a=np.array([200,300,400,500,600,700,800,900,1000,1338])
acco=np.array([84.5,89.0,85.5,89.6,89.67,87.86,89.75,89.78,89.5,88.56])
maeo=np.array([0.155,0.11,0.145,0.104,0.103,0.121,0.1025,0.102,0.105,0.114])
acc=np.array([98.0,98.67,95.0,96.0,98.0,94.86,96.5,94.67,94.8,93.73])
mae=np.array([0.02,0.034,0.05,0.04,0.02,0.051,0.035,0.053,0.052,0.062])

plt.figure(figsize=(7.5,5.5))
plt.plot(a,acc,color="red",marker='D')
plt.plot(a,acco,color="blue",marker='s',ls='dotted')
plt.suptitle("Accuracy Score of Model w.r.t. Sample Size",fontsize=18)
plt.xlabel("Sample Size",fontsize=15)
plt.ylabel("Accuracy Score (%)",fontsize=15)
plt.legend(["Modified Code","Original Code"],fontsize='x-large')
plt.show()

plt.figure(figsize=(7.5,5.5))
plt.plot(a,mae,color="red",marker='D')
plt.plot(a,maeo,color="blue",marker='s',ls='dotted')
plt.suptitle("Mean Absolute Error of Model w.r.t. Sample Size",fontsize=18)
plt.xlabel("Sample Size",fontsize=15)
plt.ylabel("Mean Absolute Error",fontsize=15)
plt.legend(["Modified Code","Original Code"],fontsize='x-large')
plt.show()