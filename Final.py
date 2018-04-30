import Exponential as ex
import ANNTrain as ann
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('eggs_new.csv')

usage = data['CPU usage [%]']

exPred, exError = ex.expo(usage)

mse_history = ann.ret_mse()
annPred = []


for i in range(len(mse_history)):
    annPred.append(usage[i] + mse_history[i])

exContri = []
annContri = []
finPred = []
enError = []
enAcc = []
axis = []
actual = []

for i in exError:
    exContri.append(1/i)

for i in mse_history:
    annContri.append(1/i)

for j in range(len(mse_history)):
    a = (annContri[j] * annPred[j])/annContri[j]
    b = (exContri[j] * exPred[j])/exContri[j]
    finPred.append((a + b)/2)

for k in range(len(finPred)):
    print('Actual = ' + str(usage[k]) + '\t\t\tPredicted = ' + str(finPred[k]))
    enError.append(abs(finPred[k] - usage[k]))
    enAcc.append((finPred[k] - usage[k]) / usage[k])
    actual.append(usage[k])
    axis.append(k)


plt.plot(axis, finPred, label='Predicted Values')
plt.plot(axis, actual, label='Actual Values')
plt.title('Predicted Values v/s Actual Values')
plt.xlabel('Iterations')
plt.ylabel('CPU Usage [%]')
plt.legend()
plt.show()

plt.plot(enError)
plt.title('Error Plot')
plt.xlabel('Iterations')
plt.ylabel('Degree of Error')
plt.show()

plt.plot(enAcc)
plt.title('Accuracy Plot')
plt.xlabel('Iterations')
plt.ylabel('Degree of Accuracy')
plt.show()
