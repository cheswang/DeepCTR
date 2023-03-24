# DeepCTR
## Environment
details docs visit: https://confluence.amd.com/display/~cheswang/DeepCTR
docker pull rocm/tensorflow:rocm5.4-tf2.10-dev;
sign in this Docker;
pip install pandas scikit-learn deepctr
git clone https://github.com/cheswang/DeepCTR.git
cd ./DeepCTR

## Run command


run DCN v2 on 1 gpu:python main.py --model_type dcnv2 --embedding_dims 64 --batch_size 6400 --num_gpus 1
run DCN v2 on 8 gpu:	python main.py --model_type dcnv2 --embedding_dims 64 --batch_size 6400 --num_gpus 8
run FwFM on 1 gpu:	 python main.py --model_type fwfm --embedding_dims 64 --batch_size 6400 --num_gpus 1
run FwFM with different embedding dimension:	 python main.py --model_type fwfm --embedding_dims 128 --batch_size 6400 --num_gpus 1




## Performance:

### model config:

Config	parameter
GPU	MI250
DNN Layer	4 layers, [1024, 512, 256, 128] respectively
Optimizer	Adam
Percision	FP32
Sparse feature dimension	
26 same as criteo_dims = [7912889,33823,582469,245828,1,2209,10667,
104,4,968,15,8165896,17139,2675940, 7156453,302516,12022,97,35,7339,20046,4,7105,1382,63,5554114]
Dense feature	13
Batch_size per GPU	6400
embedding hash	no




### DCNV2:

num_gpus	embedding_dims	parameter	throughout(examples/s)

	128	 4,225,246,304	OOM

	96	3,177,335,300	OOM
1	64	 2,129,358,448	28932.81
4	64	 2,129,358,448	52376.15
8	64	 2,129,358,448	132043.35




### FwFM:

num_gpus	embedding_dims	parameter	throughout(examples/s)

	128	 4,223,546,863	OOM

	96	3,176,008,444	OOM
1	64	2,128,453,594	21213.20
4	64	 2,128,453,594	45359.21
8	64	 2,128,453,594	102303.92 