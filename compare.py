import utils
import numpy as np
import TensorAlgebra
import struct
import time

error = 0
batchMode = utils.batchMode
xfile = open(utils.smallTensor, "rb")
dexfile = open(utils.decompTensor, "rb")
dit = utils.dit
xI, xJ, xK = struct.unpack("iii", xfile.read(12))
dI, dJ, dK = struct.unpack("iii", dexfile.read(12))
if xI != dI:
    print("alg error!")
    print(xI)
    print("while ", dI)
I = xI
J = xJ
K = xK
look = set({})
np.set_printoptions(suppress=True)
print("batchMode:{} for diff".format(batchMode))
ibatch = I // batchMode
jbatch = J // batchMode
kbatch = K // batchMode
numOfPage = (I * J * K) // (batchMode ** 3)
itemp = 0
jtemp = 0
ktemp = 0
xtensor = np.zeros([batchMode, batchMode, batchMode])
dextensor = np.zeros([batchMode, batchMode, batchMode])
total = time.time()
page = 0
mse = 0
err = 0
N = 1
loss = 0
for i in range(ibatch):
    for j in range(jbatch):
        for k in range(kbatch):
            xfile.seek(dit * i * J * K * batchMode + dit * j * K * batchMode + dit * k * batchMode + 12, 0)
            dexfile.seek(dit * i * J * K * batchMode + dit * j * K * batchMode + dit * k * batchMode + 12, 0)
            xtensor[:, :, :] = 0
            dextensor[:, :, :] = 0
            begin = time.time()
            print("page:{} begin".format(N))
            for itemp in range(batchMode):
                xfile.seek(dit * (i * batchMode + itemp) * J * K + dit * j * batchMode * K + dit * k * batchMode + 12,
                           0)
                dexfile.seek(dit * (i * batchMode + itemp) * J * K + dit * j * batchMode * K + dit * k * batchMode + 12,
                             0)
                for jtemp in range(batchMode):
                    xfile.seek(
                        dit * (i * batchMode + itemp) * J * K + dit * (
                                    j * batchMode + jtemp) * K + dit * k * batchMode + 12,
                        0)
                    dexfile.seek(
                        dit * (i * batchMode + itemp) * J * K + dit * (
                                    j * batchMode + jtemp) * K + dit * k * batchMode + 12,
                        0)
                    xcover = xfile.read(batchMode * dit)
                    dexcover = dexfile.read(batchMode * dit)
                    # print(itemp,jtemp,len(cover))
                    xtensor[itemp, jtemp, :] = np.array(struct.unpack("={}d".format(batchMode), xcover))
                    dextensor[itemp, jtemp, :] = np.array(struct.unpack("={}d".format(batchMode), dexcover))
            err = TensorAlgebra.norm(dextensor - xtensor, 2)
            # for elem in xtensor.flatten().tolist():
            # look.add(elem)
            # print(dextensor)
            print("------------------------------")
            # print(xtensor)
            mse += (1 / N) * (err - mse)
            loss += np.sum(np.power(dextensor - xtensor, 2))
            print("error in time:")
            print("batch:{}".format(N))
            print("batch MSE:{}".format(mse))
            print("batch err:{}".format(err))
            N += 1
print("total L2 loss:{} ".format(pow(loss, 1 / 2)))