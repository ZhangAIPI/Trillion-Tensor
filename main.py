import utils
import numpy as np
import core
import TensorAlgebra
import pickle as pkl
import struct
import time
import argparse

parser = argparse.ArgumentParser(description="settings")
batchMode = utils.batchMode  # 每次读取的数据的最大张量块
anchorSize = utils.anchorSize  # 用于恢复scaling的张量块的大小
parser.add_argument("--batchMode", type=int, default=batchMode, help="size of block batchMode x batchMode x batchMode")
parser.add_argument("--anchorSize", type=int, default=anchorSize, help="size of anchor")
args = parser.parse_args()
batchMode = args.batchMode
anchorSize = args.anchorSize


start = time.time()
comp = utils.compsmallv2  # 压缩文件的信息
fp = open(comp, "r")

dit = utils.dit  # 存储位数
decompfp = open(utils.decompTensor, "wb")
str = fp.readline().replace("\n", "").split(" ")
fp.close()
I, J, K, L, M, N, F, P, S = [eval(elem) for elem in str]
decompfp.write(struct.pack("iii", I, J, K))

print("number of compressed files:{}".format(P))
page = 1
Yset = []

with open("Zip/seedWithUVWv2.pkl", "rb") as sp:
    random = pkl.load(sp)
for p in range(P):
    with open("Zip/compv2-{}.pkl".format(p), "rb") as comfile:
        Yset.append(pkl.load(comfile))
# Yset存储着P个压缩后的张量 shape:PxLxMxN

compsource = [Yset, random, F, S, I, J, K]
estA, estB, estC, n = core.decomp2ABC(compsource, verbose=1)
print("get normalized components")
# print("scaling 文件共有:{} 个".format(I * J * K // (batchMode * batchMode * batchMode)))
num = I * J * K // (batchMode * batchMode * batchMode)
with open("Zip//norm.pkl".format(page), "rb") as normfile:
    anchorA, anchorB, anchorC = pkl.load(normfile)
# anchorA,anchorB,anchorC是压缩过程中为了解压缩后恢复scaling信息而对第一块张量进行分解得到的秩一分解量
"""
Method for restoring the extracted resulting estA, estB, estC to A, B, C:
    1. Normalize the anchorA, anchorB, and anchorC in the compression process by column (divide by the element with the largest modulo by column) and save the divisor of each column
    2. Take estA, estB, estC corresponds to the anchor A/B/C before the anchorSize row as blockA/B/C, and normalize by column
    3。 The normalized blockA is rearranged to the column order of the anchorA by the method with the smallest difference norm of the same column, and the column rearrangement matrix is obtained
    4. Acting on column rearrangement matrices on estA, estB, estC
    5. First normalize the estA/B/C by column of the previous anchorSize row, and then multiply by the multiplier of the corresponding column of the anchor A/B/C saved in 1 step to restore the scaling information
"""
anchornorma = []
anchornormb = []
anchornormc = []
for f in range(F):
    anorm = anchorA[:, f][0]
    bnorm = anchorB[:, f][0]
    cnorm = anchorC[:, f][0]
    for elem in anchorA[:, f]:
        if np.abs(elem) > np.abs(anorm):
            anorm = elem
    for elem in anchorB[:, f]:
        if np.abs(elem) > np.abs(bnorm):
            bnorm = elem
    for elem in anchorC[:, f]:
        if np.abs(elem) > np.abs(cnorm):
            cnorm = elem
    anchorA[:, f] = anchorA[:, f] / anorm
    anchorB[:, f] = anchorB[:, f] / bnorm
    anchorC[:, f] = anchorC[:, f] / cnorm
    anchornorma.append(anorm)
    anchornormb.append(bnorm)
    anchornormc.append(cnorm)
blocknorma = []
blocknormb = []
blocknormc = []
blockSize = anchorSize
blockA = np.zeros_like(estA[:blockSize, :])
blockB = np.zeros_like(estB[:blockSize, :])
blockC = np.zeros_like(estC[:blockSize, :])
for f in range(F):
    anorm = estA[:blockSize, f][0]
    bnorm = estB[:blockSize, f][0]
    cnorm = estC[:blockSize, f][0]
    for elem in estA[:blockSize, f]:
        if np.abs(elem) > np.abs(anorm):
            anorm = elem
    for elem in estB[:blockSize, f]:
        if np.abs(elem) > np.abs(bnorm):
            bnorm = elem
    for elem in estC[:blockSize, f]:
        if np.abs(elem) > np.abs(cnorm):
            cnorm = elem
    blockA[:blockSize, f] = estA[:blockSize, f] / anorm
    blockB[:blockSize, f] = estB[:blockSize, f] / bnorm
    blockC[:blockSize, f] = estC[:blockSize, f] / cnorm
    blocknorma.append(anorm)
    blocknormb.append(bnorm)
    blocknormc.append(cnorm)
permutaiton = core.permColmatch(anchorA, blockA)
estA = estA @ permutaiton
estB = estB @ permutaiton
estC = estC @ permutaiton
blockA = blockA @ permutaiton
blockB = blockB @ permutaiton
blockC = blockC @ permutaiton

for f in range(F):
    blockA[:, f] = blockA[:, f] * anchornorma[f]
    blockB[:, f] = blockB[:, f] * anchornormb[f]
    blockC[:, f] = blockC[:, f] * anchornormc[f]
    anchorA[:, f] = anchorA[:, f] * anchornorma[f]
    anchorB[:, f] = anchorB[:, f] * anchornormb[f]
    anchorC[:, f] = anchorC[:, f] * anchornormc[f]
"""
The output prints the recovery L2 error of the small tensor block corresponding to anchor A/B/C
"""
print("check here:{}".format(
    TensorAlgebra.norm(TensorAlgebra.kruskal_to_tensor([blockA, blockB, blockC]) - TensorAlgebra.kruskal_to_tensor(
        [anchorA, anchorB, anchorC]), order=2)))

for f in range(F):
    anorm = estA[:blockSize, f][0]
    bnorm = estB[:blockSize, f][0]
    cnorm = estC[:blockSize, f][0]
    for elem in estA[:blockSize, f]:
        if np.abs(elem) > np.abs(anorm):
            anorm = elem
    for elem in estB[:blockSize, f]:
        if np.abs(elem) > np.abs(bnorm):
            bnorm = elem
    for elem in estC[:blockSize, f]:
        if np.abs(elem) > np.abs(cnorm):
            cnorm = elem

    estA[:, f] = estA[:, f] * anchornorma[f] / anorm
    estB[:, f] = estB[:, f] * anchornormb[f] / bnorm
    estC[:, f] = estC[:, f] * anchornormc[f] / cnorm
next = 0
for page in range(1):

    for i in range(I // batchMode):
        tempA = np.zeros([batchMode, F])
        for j in range(J // batchMode):
            tempB = np.zeros([batchMode, F])
            for k in range(K // batchMode):
                # tempA = np.zeros([batchMode, F])
                # tempB = np.zeros([batchMode, F])
                tempC = np.zeros([batchMode, F])
                next += 1
                print("recovering block:{} ".format(next))
                for f in range(F):
                    tempA[:, f] = estA[i * batchMode:(i + 1) * batchMode, f]  # * nA[f]
                    tempB[:, f] = estB[j * batchMode:(j + 1) * batchMode, f]  # * nB[f]
                    tempC[:, f] = estC[k * batchMode:(k + 1) * batchMode, f]  # * nC[f]
                tempX = TensorAlgebra.kruskal_to_tensor([tempA, tempB, tempC])
                for itemp in range(batchMode):
                    decompfp.seek(
                        dit * (i * batchMode + itemp) * J * K + dit * j * batchMode * K + dit * k * batchMode + 12, 0)
                    for jtemp in range(batchMode):
                        decompfp.seek(dit * (i * batchMode + itemp) * J * K + dit * (
                                j * batchMode + jtemp) * K + dit * k * batchMode + 12, 0)
                        decompfp.write(struct.pack("={}d".format(batchMode), *tempX[itemp, jtemp, :]))

print("data saved")
decompfp.close()
print("time used:{} s".format(time.time() - start))