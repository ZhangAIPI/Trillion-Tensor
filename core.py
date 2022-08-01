import TensorAlgebra
from scipy.optimize import linear_sum_assignment
from scipy.linalg import lstsq
import numpy as np
import pickle
from multiprocessing import Pool


def permColmatch(iA, dA):
    """
    该方法依靠匈牙利算法，寻找一个列重排矩阵pai使得 L2 norm ||iA-dA@pai|| 尽可能小
    :param iA:目标矩阵
    :param dA: 待匹配矩阵
    :return: 列重排矩阵
    """
    iA = np.array(iA)
    dA = np.array(dA)
    pre = iA.T @ dA
    # print("permColmatch before trace:{}".format(np.trace(pre)))
    cost = np.max(pre) - pre
    LAProw, LAPcol = linear_sum_assignment(cost)
    _, F = iA.shape
    zeors = np.zeros([F, F])
    for row, col in zip(LAProw, LAPcol):
        zeors[row, col] = 1
    zeors = zeors.T
    # print("permColmatch after trace:{}".format(np.trace(pre @ zeors)))
    print("after perm anchor difference L2 norm:{}".format(np.linalg.norm(iA - dA @ zeors)))
    return zeors


def paraMultyply(tensor, random, shape):
    U, V, W = random
    # print(U.shape)
    I, J, K, L, M, N = shape
    G1 = np.reshape(tensor, newshape=[I, J * K])
    G2 = np.reshape(U.T.__matmul__(G1), newshape=[L * J, K]).T
    G3 = np.reshape(W.T.__matmul__(G2), newshape=[L * N, J]).T
    G4 = np.reshape(V.T.__matmul__(G3), newshape=[M * N, L]).T
    return np.reshape(G4, newshape=[L, M, N])


def paraComp(tensor, compSize, P=12, S=3, F=5, seed=None, n_iter_max=5000, tol=1e-15, verbose=0):
    if seed == None:
        seed = [np.random.randint(0, 10000, size=3) for i in range(P)]
    if type(tensor) is not np.ndarray:
        tensor = np.array(tensor)
    I, J, K = tensor.shape
    L, M, N = compSize
    """Uset = []
    Vset = []
    Wset = []"""
    Yset = []
    _, Uset, Vset, Wset = seed
    for p in range(P):
        """ # prepare the U/V/W random matrix and set the common S cols anchor
         np.random.seed(seed[p][0])
         Uset.append(np.random.normal(size=[I, L]))
         Uset[p][:, :S] = Uset[0][:, :S]
         np.random.seed(seed[p][1])
         Vset.append(np.random.normal(size=[J, M]))
         Vset[p][:, :S] = Vset[0][:, :S]
         np.random.seed(seed[p][2])
         Wset.append(np.random.normal(size=[K, N]))
         Wset[p][:, :S] = Wset[0][:, :S]"""
        # compress the Tensor
        """G1 = np.reshape(tensor, newshape=[I, J * K])
        G2 = np.reshape(Uset[p].T.__matmul__(G1), newshape=[L * J, K]).T
        G3 = np.reshape(Wset[p].T.__matmul__(G2), newshape=[L * N, J]).T
        G4 = np.reshape(Vset[p].T.__matmul__(G3), newshape=[M * N, L]).T"""
        G4 = paraMultyply(tensor, [Uset[p], Vset[p], Wset[p]], [I, J, K, L, M, N])
        Yset.append(G4)
    Yset = np.array(Yset)
    normColA = []
    normColB = []
    normColC = []
    [A, B, C] = TensorAlgebra.parafac(tensor, rank=F, n_iter_max=n_iter_max, tol=tol, verbose=verbose)

    for f in range(F):
        a = A[:, f][0]
        b = B[:, f][0]
        c = C[:, f][0]
        for elem in A[:, f]:
            if np.abs(elem) > np.abs(a):
                a = elem
        for elem in B[:, f]:
            if np.abs(elem) > np.abs(b):
                b = elem
        for elem in C[:, f]:
            if np.abs(elem) > np.abs(c):
                c = elem
        normColA.append(a)
        normColB.append(b)
        normColC.append(c)
        print(a)
        print(b)
        print(c)
    comsource = [Yset, [normColA, normColB, normColC], seed, S, I, J, K]
    return comsource


def decomp(compsource, verbose=0):
    Yset, norm, seed, S, I, J, K = compsource
    P = len(seed[0])
    F = len(norm[0])
    Yset = np.array(Yset)
    L, M, N = Yset[0].shape
    """Uset = []
    Vset = []
    Wset = []"""
    _, Uset, Vset, Wset = seed
    anchor = []
    Y2Aset = []
    Y2Bset = []
    Y2Cset = []
    rowStackU = None
    rowStackV = None
    rowStackW = None
    rowStackA = None
    rowStackB = None
    rowStackC = None
    print(P)
    for p in range(P):
        # prepare the U/V/W random matrix and set the common S cols anchor
        """np.random.seed(seed[p][0])
        Uset.append(np.random.normal(size=[I, L]))
        Uset[p][:, :S] = Uset[0][:, :S]
        np.random.seed(seed[p][1])
        Vset.append(np.random.normal(size=[J, M]))
        Vset[p][:, :S] = Vset[0][:, :S]
        np.random.seed(seed[p][2])
        Wset.append(np.random.normal(size=[K, N]))
        Wset[p][:, :S] = Wset[0][:, :S]"""
        [facA, facB, facC] = TensorAlgebra.parafac(Yset[p], tol=1e-15, n_iter_max=30000, rank=F, verbose=verbose)
        Y2Aset.append(np.asarray(facA))
        Y2Bset.append(np.asarray(facB))
        Y2Cset.append(np.asarray(facC))
    permutationSet = []
    anchorOffacA = np.array([Y2Aset[i][:S, :] for i in range(P)])
    # devided by  the maximum module
    for p in range(P):
        for col in range(anchorOffacA[p].shape[1]):
            maxModule = anchorOffacA[p][:, col][0]
            for elem in anchorOffacA[p][:, col]:
                if np.abs(elem) > np.abs(maxModule):
                    maxModule = elem

            # print(maxModule)
            anchorOffacA[p][:, col] /= maxModule
        # print("check divided max modules")
        # print(anchorOffacA[p])
        # print("--------------")

    # print(type(anchorOffacA[0]))
    costMat = np.array([anchorOffacA[0].T.__matmul__(anchorOffacA[p]) for p in range(P)])
    # print(type(costMat))
    facAafterPermutation = []
    facBafterPermutation = []
    facCafterPermutation = []
    mincostMat = []
    for p in range(P):
        cost = np.array(costMat[p])
        cost = cost.max() - cost
        # print("cost.max :{} in  p ".format(cost))
        mincostMat.append(cost)
        # print("before LAP P:{} trace:{} ".format(p,np.trace(costMat[p])))
        LAProw, LAPcol = linear_sum_assignment(cost)

        zeors = np.zeros([F, F])
        for row, col in zip(LAProw, LAPcol):
            zeors[row, col] = 1
        zeors = zeors.T
        permutationSet.append(zeors)
        facAafterPermutation.append(Y2Aset[p].__matmul__(zeors))
        facBafterPermutation.append(Y2Bset[p].__matmul__(zeors))
        facCafterPermutation.append(Y2Cset[p].__matmul__(zeors))

    # normalize
    isNormalize = True
    normalizeSet = []
    if isNormalize:
        for p in range(P):
            for col in range(F):
                anorm = facAafterPermutation[p][:, col][0]
                bnorm = facBafterPermutation[p][:, col][0]
                cnorm = facCafterPermutation[p][:, col][0]
                for elem in facAafterPermutation[p][:S, col]:
                    if np.abs(elem) > np.abs(anorm):
                        anorm = elem
                for elem in facBafterPermutation[p][:S, col]:
                    if np.abs(elem) > np.abs(bnorm):
                        bnorm = elem
                for elem in facCafterPermutation[p][:S, col]:
                    if np.abs(elem) > np.abs(cnorm):
                        cnorm = elem
                # anorm = np.linalg.norm(facAafterPermutation[p][:S, col])

                facAafterPermutation[p][:, col] /= anorm
                facBafterPermutation[p][:, col] /= bnorm
                facCafterPermutation[p][:, col] /= cnorm
    rowStackA = np.array(facAafterPermutation[0])
    rowStackB = np.array(facBafterPermutation[0])
    rowStackC = np.array(facCafterPermutation[0])
    rowTstackU = np.array(Uset[0].T)
    rowTstackV = np.array(Vset[0].T)
    rowTstackW = np.array(Wset[0].T)
    for p in range(1, P):
        rowStackA = np.row_stack([rowStackA, facAafterPermutation[p]])
        rowStackB = np.row_stack([rowStackB, facBafterPermutation[p]])
        rowStackC = np.row_stack([rowStackC, facCafterPermutation[p]])
        rowTstackU = np.row_stack([rowTstackU, Uset[p].T])
        rowTstackV = np.row_stack([rowTstackV, Vset[p].T])
        rowTstackW = np.row_stack([rowTstackW, Wset[p].T])

    yset = [rowStackA, rowStackB, rowStackC]
    Xset = [rowTstackU, rowTstackV, rowTstackW]

    estimateA = np.linalg.pinv(Xset[0]).__matmul__(yset[0])
    estimateB = np.linalg.pinv(Xset[1]).__matmul__(yset[1])
    estimateC = np.linalg.pinv(Xset[2]).__matmul__(yset[2])

    for f in range(F):
        anorm = estimateA[:, f][0]
        bnorm = estimateB[:, f][0]
        cnorm = estimateC[:, f][0]
        for elem in estimateA[:, f]:
            if np.abs(elem) > np.abs(anorm):
                anorm = elem
        for elem in estimateB[:, f]:
            if np.abs(elem) > np.abs(bnorm):
                bnorm = elem
        for elem in estimateC[:, f]:
            if np.abs(elem) > np.abs(cnorm):
                cnorm = elem
        estimateA[:, f] = estimateA[:, f] / anorm * norm[0][f]
        estimateB[:, f] = estimateB[:, f] / bnorm * norm[1][f]
        estimateC[:, f] = estimateC[:, f] / cnorm * norm[2][f]

    estimateX = TensorAlgebra.kruskal_to_tensor([estimateA, estimateB, estimateC])

    return estimateX


def decomp2ABC(compsource, verbose=0):
    """
    :param compsource:[Yset,seed,F,S,I,J,K]
    Yset:shape=PxLxMxN stores P compression tensors to be extracted
    seed: [seed, Uset, Vset, Wset] stores P positive distribution matrices for random compression
    :p aram verbose: Shows the progress of cp decomposition
    :return: Normalized A/B/C
    """
    Yset, seed, F, S, I, J, K = compsource
    P = len(seed[0])
    Yset = np.array(Yset).astype(np.float64)
    _, Uset, Vset, Wset = seed
    # Store the A/B/C components corresponding to each P compression matrix separately
    Y2Aset = []
    Y2Bset = []
    Y2Cset = []

    # 将每个压缩矩阵进行cp分解并对应存储到列表中
    for p in range(P):
        [facA, facB, facC] = TensorAlgebra.parafac(Yset[p].astype(np.float64), tol=1e-15, n_iter_max=30000, rank=F,
                                                   verbose=verbose)
        Y2Aset.append(np.asarray(facA))
        Y2Bset.append(np.asarray(facB))
        Y2Cset.append(np.asarray(facC))
    # Each compression matrix is cp-decomposed and stored in the corresponding list
    permutationSet = []
    anchorOffacA = np.array([Y2Aset[i][:S, :] for i in range(P)])

    # Normalize the public matrix by column devided by the maximum module
    for p in range(P):
        for col in range(anchorOffacA[p].shape[1]):
            maxModule = anchorOffacA[p][:, col][0]
            for elem in anchorOffacA[p][:, col]:
                if np.abs(elem) > np.abs(maxModule):
                    maxModule = elem
            anchorOffacA[p][:, col] /= maxModule

    # Separately find the cost matrix of the Hungarian algorithm for each P corresponding to the A component
    costMat = np.array([anchorOffacA[0].T.__matmul__(anchorOffacA[p]) for p in range(P)])

    # The following three lists store the A/B/C components of a column rearrangement (eliminating the effects of columns).
    facAafterPermutation = []
    facBafterPermutation = []
    facCafterPermutation = []
    mincostMat = []
    for p in range(P):
        cost = np.array(costMat[p])
        cost = cost.max() - cost
        # print("cost.max :{} in  p ".format(cost))
        mincostMat.append(cost)
        # print("before LAP P:{} trace:{} ".format(p,np.trace(costMat[p])))
        LAProw, LAPcol = linear_sum_assignment(cost)

        zeors = np.zeros([F, F])
        for row, col in zip(LAProw, LAPcol):
            zeors[row, col] = 1
        zeors = zeors.T
        permutationSet.append(zeors)

        # Eliminate the influence of A/B/C columns and unify to the column order of p=1
        facAafterPermutation.append(Y2Aset[p].__matmul__(zeors))
        facBafterPermutation.append(Y2Bset[p].__matmul__(zeors))
        facCafterPermutation.append(Y2Cset[p].__matmul__(zeors))

    # Normalize normalizes the ABC component again through the front S line
    isNormalize = True
    normalizeSet = []
    if isNormalize:
        for p in range(P):
            for col in range(F):
                anorm = facAafterPermutation[p][:, col][0]
                bnorm = facBafterPermutation[p][:, col][0]
                cnorm = facCafterPermutation[p][:, col][0]
                for elem in facAafterPermutation[p][:S, col]:
                    if np.abs(elem) > np.abs(anorm):
                        anorm = elem
                for elem in facBafterPermutation[p][:S, col]:
                    if np.abs(elem) > np.abs(bnorm):
                        bnorm = elem
                for elem in facCafterPermutation[p][:S, col]:
                    if np.abs(elem) > np.abs(cnorm):
                        cnorm = elem
                # anorm = np.linalg.norm(facAafterPermutation[p][:S, col])

                facAafterPermutation[p][:, col] /= anorm
                facBafterPermutation[p][:, col] /= bnorm
                facCafterPermutation[p][:, col] /= cnorm
    # Construct the coefficients of a linear equation
    # rowStackA/B/C=rowTstackU/V/W@estA/B/C
    # estA/B/C is the component of the unmerized original tensor sought
    rowStackA = np.array(facAafterPermutation[0])
    rowStackB = np.array(facBafterPermutation[0])
    rowStackC = np.array(facCafterPermutation[0])
    rowTstackU = np.array(Uset[0].T)
    rowTstackV = np.array(Vset[0].T)
    rowTstackW = np.array(Wset[0].T)
    for p in range(1, P):
        rowStackA = np.row_stack([rowStackA, facAafterPermutation[p]])
        rowStackB = np.row_stack([rowStackB, facBafterPermutation[p]])
        rowStackC = np.row_stack([rowStackC, facCafterPermutation[p]])
        rowTstackU = np.row_stack([rowTstackU, Uset[p].T])
        rowTstackV = np.row_stack([rowTstackV, Vset[p].T])
        rowTstackW = np.row_stack([rowTstackW, Wset[p].T])

    yset = [rowStackA, rowStackB, rowStackC]
    Xset = [rowTstackU, rowTstackV, rowTstackW]

    estimateA = np.linalg.pinv(Xset[0]).__matmul__(yset[0])
    estimateB = np.linalg.pinv(Xset[1]).__matmul__(yset[1])
    estimateC = np.linalg.pinv(Xset[2]).__matmul__(yset[2])

    print("----------------------------------------------------------")
    print("pinv check: L2-norm")
    print("a check:{}".format(np.linalg.norm(Xset[0] @ estimateA - yset[0])))
    print("b check:{}".format(np.linalg.norm(Xset[1] @ estimateB - yset[1])))
    print("c check:{}".format(np.linalg.norm(Xset[2] @ estimateC - yset[2])))
    print("-----------------------------------------------------------")
    na = []
    nb = []
    nc = []
    # The resulting A/B/C is normalized by column
    for f in range(F):
        anorm = estimateA[:, f][0]
        bnorm = estimateB[:, f][0]
        cnorm = estimateC[:, f][0]
        for elem in estimateA[:, f]:
            if np.abs(elem) > np.abs(anorm):
                anorm = elem
        for elem in estimateB[:, f]:
            if np.abs(elem) > np.abs(bnorm):
                bnorm = elem
        for elem in estimateC[:, f]:
            if np.abs(elem) > np.abs(cnorm):
                cnorm = elem
        estimateA[:, f] = estimateA[:, f] / anorm
        estimateB[:, f] = estimateB[:, f] / bnorm
        estimateC[:, f] = estimateC[:, f] / cnorm
        na.append(anorm)
        nb.append(bnorm)
        nc.append(cnorm)
    return [estimateA, estimateB, estimateC, [na, nb, nc]]