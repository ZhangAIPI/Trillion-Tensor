import numpy as np
import utils
import sys
import core
import TensorAlgebra
import struct
import argparse

parser = argparse.ArgumentParser(description="超参设置")
I = utils.I
J = utils.J
K = utils.K
F = utils.F
dit = utils.dit
outFile = utils.smallTensor
isInteger = utils.isInteger
batchMode = utils.batchMode
parser.add_argument("--I", type=int, default=I, help="size of tensor , I")
parser.add_argument("--J", type=int, default=J, help="size of tensor , J")
parser.add_argument("--K", type=int, default=K, help="size of tensor , K")
parser.add_argument("--F", type=int, default=F, help="rank of tensor")
parser.add_argument("--isInteger", type=bool, default=isInteger, help="whether generate tensors with only int")
parser.add_argument("--batchMode", type=int, default=batchMode, help="maximum size for reading/generating tenosor, batchMode x batchMode x batchMode")
args = parser.parse_args()
I = args.I
J = args.J
K = args.K
F = args.F
isInteger = args.isInteger
batchMode = args.batchMode

if not isInteger:
    A = np.random.normal(size=[I, F]).astype(
        float)
    while np.linalg.matrix_rank(A) != F:
        A = np.random.normal(size=[I, F]).astype(
            float)

    B = np.random.normal(size=[J, F]).astype(
        float)
    while np.linalg.matrix_rank(B) != F:
        B = np.random.normal(size=[J, F]).astype(
            float)

    C = np.random.normal(size=[K, F]).astype(
        float)
    while np.linalg.matrix_rank(C) != F:
        C = np.random.normal(size=[K, F]).astype(
            float)
else:
    A = np.random.randint(low=-10, high=10, size=[I, F]).astype(
        float)
    while np.linalg.matrix_rank(A) != F:
        A = np.random.randint(low=-10, high=10, size=[I, F]).astype(
            float)

    B = np.random.randint(low=-10, high=10, size=[J, F]).astype(
        float)
    while np.linalg.matrix_rank(B) != F:
        B = np.random.randint(low=-10, high=10, size=[J, F]).astype(
            float)

    C = np.random.randint(low=-10, high=10, size=[K, F]).astype(
        float)
    while np.linalg.matrix_rank(C) != F:
        C = np.random.randint(low=-10, high=10, size=[K, F]).astype(
            float)

info = "{} {} {} \n".format(I, J, K)

file = open(outFile, "wb")

file.write(struct.pack("iii", I, J, K))

tempA = np.zeros([batchMode, F])
tempB = np.zeros([batchMode, F])
tempC = np.zeros([batchMode, F])
next = 0
"""
save"""

for i in range(I // batchMode):
    for j in range(J // batchMode):
        for k in range(K // batchMode):
            next += 1
            print("writing block {}-th".format(next))
            for f in range(F):
                tempA[:, f] = A[i * batchMode:(i + 1) * batchMode, f]
                tempB[:, f] = B[j * batchMode:(j + 1) * batchMode, f]
                tempC[:, f] = C[k * batchMode:(k + 1) * batchMode, f]
            tempX = TensorAlgebra.kruskal_to_tensor([tempA, tempB, tempC])
            for itemp in range(batchMode):
                for jtemp in range(batchMode):
                    file.seek(dit * (i * batchMode + itemp) * J * K + dit * (
                            j * batchMode + jtemp) * K + dit * k * batchMode + 12, 0)
                    file.write(struct.pack("={}d".format(batchMode), *tempX[itemp, jtemp, :]))

print("save ok!")

file.close()