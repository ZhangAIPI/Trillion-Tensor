import numpy as np
import core
import TensorAlgebra
import utils
import time
import struct
import argparse

parser = argparse.ArgumentParser(description="setting the hyper-parameter")
# the config info is saved in utils.py
L = utils.L
M = utils.M
N = utils.N
dit = utils.dit  # number of digits to save the tensor (e.g. 8, 16)
S = utils.S  # anchor  size to recover the permutation
P=utils.P
F = utils.F
batchMode = utils.batchMode  # input Max Tensor Core from I/O
anchorSize = utils.anchorSize

parser.add_argument("--batchMode", type=int, default=batchMode, help="the maximum size to read the tensor batchMode x batchMode x batchMode")
parser.add_argument("--anchorSize", type=int, default=anchorSize, help="anchor tensor for recovering the  scaling")
parser.add_argument("--L", type=int, default=L, help="dim L of the compressed tensor")
parser.add_argument("--M", type=int, default=M, help="dim M of the compressed tensor")
parser.add_argument("--N", type=int, default=N, help="dim N of the compressed tensor")
parser.add_argument("--S", type=int, default=S, help="number of columns in anchor tenosr")
parser.add_argument("--P", type=int, default=None, help="number of paralle compressed tensors")

args = parser.parse_args()
batchMode = args.batchMode
anchorSize = args.anchorSize
L = args.L
M = args.L
N = args.N
S = args.S

# save the compressed tensor in utils.smallTensor
fp = open(utils.smallTensor, "rb")
info = fp.read(12)  # I,J,K
shapeInfo = struct.unpack("iii", info)
I, J, K = shapeInfo

if args.P==None:
    P=max((I - 2) // (L - 2), I // L) + 3
else:
    P=args.P

print("info the tenosr to be compressed :\n{} x {} x {} \n {}个 {} x {} x {}的张量".format(I, J, K, P, L, M, N))
ibatch = I // batchMode  # the maximum step size in I dimension
jbatch = J // batchMode  # the maximum step size in J dimension
kbatch = K // batchMode  # the maximum step size in K dimension
page = 0
# random seed for generating the U,V,W
seed = [np.random.randint(0, 1000, size=3) for i in range(P)]

allstart = time.time()
Uset = []  
Vset = []  
Wset = [] 
embeddingY = np.zeros([P, L, M, N])  # 存储最终的P个压缩张量

for p in range(P):
    """
    generate Up, Vp, Wp and set the shared the columns used 
    """
    np.random.seed(seed[p][0])
    # np.random.seed(10)
    Uset.append(np.random.multivariate_normal([0] * L, np.diag([1] * L), I))
    Uset[p][:, :S] = Uset[0][:, :S]  # the shared columns are the first S columns
    np.random.seed(seed[p][1])
    # np.random.seed(13)
    Vset.append(np.random.multivariate_normal([0] * M, np.diag([1] * M), J))
    Vset[p][:, :S] = Vset[0][:, :S]
    np.random.seed(seed[p][2])
    # np.random.seed(17)
    Wset.append(np.random.multivariate_normal([0] * N, np.diag([1] * N), K))
    Wset[p][:, :S] = Wset[0][:, :S]
info = "{} {} {} {} {} {} {} {} {}\n".format(I, J, K, L, M, N, F, P, S)
compfp = open(utils.compsmallv2, "w")  
compfp.write(info)
compfp.close()
import pickle

with open("Zip/seedWithUVWv2.pkl", "wb") as sp:
    pickle.dump([seed, Uset, Vset, Wset], sp)  
init = time.time()

if True:
    i = 0
    j = 0
    k = 0
    anchorTensor = np.zeros([anchorSize, anchorSize, anchorSize])
    for itemp in range(anchorSize):
        fp.seek(dit * (i * anchorSize + itemp) * J * K + dit * j * anchorSize * K + dit * k * anchorSize + 12,
                0)
        for jtemp in range(anchorSize):
            fp.seek(dit * (i * anchorSize + itemp) * J * K + dit * (
                    j * anchorSize + jtemp) * K + dit * k * anchorSize + 12, 0)
            cover = fp.read(anchorSize * dit)
            anchorTensor[itemp, jtemp, :] = np.array(struct.unpack("={}d".format(anchorSize), cover))
    print("saving scaling info")
    normA, normB, normC = TensorAlgebra.parafac(anchorTensor, F, 30000, tol=10e-15, verbose=1)
    with open("Zip//norm.pkl", "wb") as normfile:
        pickle.dump([normA, normB, normC], normfile)
    print("save scaling info  ok!")
preparedtime = time.time() - allstart


tensor = np.zeros(shape=[batchMode, batchMode, batchMode])
total = time.time()
page = 0

blocktime = 0
totalblocktime = 0
ptime = 0
totalptime = 0
for i in range(ibatch):
    for j in range(jbatch):
        for k in range(kbatch):
            tensor[:, :, :] = 0
            begin = time.time()
            print("-------------------------------------")
            print("block:{} begin".format(page + 1))
            blockstart = time.time()
            for itemp in range(batchMode):
                for jtemp in range(batchMode):
                    fp.seek(dit * (i * batchMode + itemp) * J * K + dit * (
                            j * batchMode + jtemp) * K + dit * k * batchMode + 12, 0)
                    cover = fp.read(batchMode * dit)
                    tensor[itemp, jtemp, :] = np.array(struct.unpack("={}d".format(batchMode), cover))
            page += 1
            blocktime = time.time() - blockstart
            totalblocktime += blocktime
            # to compress
            for pIndex in range(P):
                pstart = time.time()
                tempU = Uset[pIndex][i * batchMode:(i + 1) * batchMode, :]
                tempV = Vset[pIndex][j * batchMode:(j + 1) * batchMode, :]
                tempW = Wset[pIndex][k * batchMode:(k + 1) * batchMode, :]
                G1 = np.reshape(tensor, newshape=[batchMode, batchMode * batchMode])
                G2 = np.reshape(tempU.T @ G1, newshape=[L * batchMode, batchMode]).T
                G3 = np.reshape(tempW.T @ G2, newshape=[L * N, batchMode]).T
                G4 = np.reshape(tempV.T @ G3, newshape=[M * N, L]).T
                embeddingY[pIndex, :, :, :] += np.reshape(G4, newshape=[L, M, N])
                ptime = time.time() - pstart
                totalptime += ptime
            print("time for reading tensor:{} s ".format(blocktime))
            print("avg time for reading:{} s".format(totalblocktime / page))
            meanblock = totalblocktime / page
            print("totoal time for reading:{} s".format(totalblocktime))
            print("time for compressing last tensor:{} s ".format(ptime))
            print("avg time for compressing:{} s ".format(totalptime / (page * P)))
            meanptime = totalptime / (page * P)
            print("total time for compressing:{} s ".format(totalptime))
            print("estimated remaining time:{} min".format(
                (meanblock * ibatch * jbatch * kbatch + meanptime * ibatch * jbatch * kbatch * P - meanblock * page + meanptime * page * P + preparedtime) / 60))
for pIndex in range(P):  # 将压缩文件进行储存
    pfile = open("Zip/compv2-{}.pkl".format(pIndex), "wb")

    pickle.dump(embeddingY[pIndex].astype(np.float64), pfile)
    # print(type(compTensor.astype(np.float16)[0,0,0]))
    print("mode-{} save!".format(pIndex))
    # print("used:{} s".format(time.time() - start))
    pfile.close()

print("total time:{} s ".format((time.time() - init) / 60))