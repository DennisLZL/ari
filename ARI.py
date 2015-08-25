__author__ = 'dennis'

import numpy as np
from scipy.misc import comb

def ARI(trueLab, predLab):
    """
    compute adjusted rand index, ranges in [-1, 1] with random assignment score 0 and perfect score 1
    :param trueLab: ground truth labels
    :param predLab: predicted labels
    :return: adjusted rand index
    """
    n = len(trueLab)
    trueLab = np.array(trueLab)
    predLab = np.array(predLab)

    trueCluster = dict(zip(set(trueLab), [np.where(trueLab == x)[0] for x in set(trueLab)]))
    predCluster = dict(zip(set(predLab), [np.where(predLab == x)[0] for x in set(predLab)]))

    nTrue = len(trueCluster)
    nPred = len(predCluster)

    cTable = np.zeros((nTrue, nPred))

    for i in range(nTrue):
        for j in range(nPred):
            cTable[i, j] = len(np.intersect1d(trueCluster.values()[i], predCluster.values()[j]))

    a = comb(np.sum(cTable, axis=1), 2).sum()
    b = comb(np.sum(cTable, axis=0), 2).sum()
    c = comb(n, 2)

    return (comb(cTable, 2).sum() - (a * b) / c) / (0.5 * (a + b) - (a * b) / c)

if __name__ == '__main__':
    trueLab = [0, 0, 0, 1, 1, 1]
    predLab = [0, 0, 1, 1, 2, 2]

    print ARI(trueLab, predLab)

    labels_true = [0, 1, 2, 0, 3, 4, 5, 1]
    labels_pred = [1, 1, 0, 0, 2, 2, 2, 2]

    print ARI(labels_pred, labels_true)


