import pandas as pd
import matplotlib.pyplot as plt

pred = []
error = []
mse = []
alpha = 0.5


def calcMSE(error):
    se = 0
    for i in error:
        se = i * i
    return se / len(error)


def expo(usage):
    for i in range(len(usage)):
        if i == 0:
            pred.append(usage[i])
        else:

            pred.append(pred[i-1] + (alpha * (usage[i-1] - pred[i-1])))
            err = abs(usage[i] - pred[i])
            error.append(err)
            mse.append(calcMSE(error))
            #print 'Actual - ' + str(usage[i]) + ' \t\t\tPrediction - ' + str(pred[i]) + ' \t\t\tError - ' + str((usage[i] - pred[i]))

    #print'\n Mean Sqaure Error - ' + str(mse[len(mse)-1])
    plt.plot(error)
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.show()

    return pred, error



#count = 0
#val = 0
#mse = []
#for i in range(len(usage)):
#    if i == 0:
#        pred.append(usage[i])
#        count += 1
#    else:
#        for j in range(count):
#            val += alpha * pow((1-alpha), j+1) * usage[i-j]
#        pred.append(val)
#        val = 0
#    error.append(usage[i] - pred[i])
#    mse.append(calcMSE(error))
#    print 'Actual - ' + str(usage[i]) + ' Prediction - ' + str(pred[i]) + ' Error - ' + str((usage[i] - pred[i]))


    #print'\n Mean Sqaure Error - ' + str(mse[len(mse)-1])
    #plt.plot(error)
    #plt.show()
