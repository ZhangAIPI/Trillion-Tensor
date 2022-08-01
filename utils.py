smallTensor = "small.txt"  # path to save compressed tensor
decompTensor = "decomp.txt"  # path to save decomposed factors
compsmallv2 = "Zip/compsmallv2.txt"  # path to save info
# number of unit bits to save
dit = 8
# size of generated tensor
I = 200
J = 200
K = 200
# size of compressed tensor
L = 50
M = 50
N = 50
batchMode = 100  # size of processed block batchMode*batchMode*batchMode
anchorSize = 50  # size of block used for recovery stage anchorSize*anchorSize*anchoSize
F = 5
P = max((I - 2) // (L - 2), I // L) + 3
S = 3  # number of shared columns 

isInteger = False  # whether to generate normalize data