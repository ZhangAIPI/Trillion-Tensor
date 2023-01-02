# Trillion-Tensor: Trillion-Scale CP Tensor Decomposition  


## Abstract    
Due to the storage limitation, tensors with billions or trillions of nonzero elements cannot be loaded into the main memory. Existing tensor decomposition methods only support billion-scale tensors. In this paper, we implement a compression-based algorithm trillion-tensor for trillion-scale (number of elements) CP tensor decomposition by trading computation for storage.  We make full use of the parallelism of tensor operations to accelerate the proposed trillion-tensor scheme on CPUs and GPUs, respectively. In our experiments, we test tensors ranging from million-scale to trillion-scale and obtain a relatively low mean squared error.  Comparing to the baseline method PARACOMP, trillion-tensor supports 8,000 larger tensors and a speedup up to 6.95x.    

## Overview  
![image](figs/modify.jpg)  
An example for our parallel Trillion-Tensor scheme. 


## How to Use  
utils.py: basic configure
TensorAlgebra.py: basic tensor algebra operations  
core.py: core implementation of trillion-tensor  
main.py: a test case for trillion-tensor    (also the decompress file)
generate.py: generate large scale tensors, of which information is suppied by utils.py  
compare.py:  evaluate the performance of our method (L2 reconstruction error)  
compress.py: compress the large scale tensor  

pipeline: write the config in utils.py -> use generate.py to generate large scale tensor -> use compress.py to compress the tensor -> use main.py to decompose the compressed tensor  -> use compare.py to evaluate the performance.  

If you have any problem when running our code, please feel free to concat Zeliang (email: hust0426@gmail.com)  

## Citation  
If you find Trillion-Tensor useful in your research, please consider citing:  
@article{Zhang2020trillion,  
    Author = {Zeliang Zhang, Xiao-Yang Liu, Pan Zhou},  
    Title = {Trillion-Tensor: Trillion-Scale CP Tensor Decomposition},  
    Journal = {IJCAI 2020 Workshop on Tensor Network Represenations in Machine Learning},   
    Year = {2020}   
}     
Our paper can be accessed online from https://tensorworkshop.github.io/2020/accepted_papers/4-Trillion-Tensor-Trillion-Scale%20CP%20Tensor%20Decomposition.pdf  
