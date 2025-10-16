**This version is modified for use with AAUs AI-LAB**
AI-LAB is a cluster which is slurm based. More information can be found at: https://hpc.aau.dk/ai-lab/
To access AI-LAB it is necessary to be on the same network or use a vpn to the network.
A guide for this can be found at: https://www.en.its.aau.dk/instructions/VPN#set-up-vpn-on-windows-and-mac-


Some files to explore:
-  train_percepnet.sh # This file is used for setting up the training as a task in AI-LAB
-  train_percepnet_with_requeuing.sh # This adds requeueing to the above script, which is necessary for longer training times
-  train_percepnet_with_tensorboard.sh # An attempt unsuccesfull attempt at adding tensorboard to the above script.
-  deploy_simple # Performs step 6 and 7 in the build guide. It dumps the trained weights into a c++ header and runs the inference.
-  PercepNet_Container.def # This is the Container/Docker used. It contains the used libraries, compilers, etc.
-  visualize_training.sh # Since the attempt at using tensorboard was unsuccesfull, a short script to visualize the training results was constructed. This runs that script.

Remember to use sbatch when starting jobs, as it does not require you to maintain a connection to the HPC while it is training.


# PercepNet
Unofficial implementation of PercepNet: A Perceptually-Motivated Approach for Low-Complexity, Real-Time Enhancement of Fullband Speech described in https://arxiv.org/abs/2008.04259

https://www.researchgate.net/publication/343568932_A_Perceptually-Motivated_Approach_for_Low-Complexity_Real-Time_Enhancement_of_Fullband_Speech

## Todo

- [X] pitch estimation
- [X] Comb filter
- [X] ERBBand c++ implementation
- [X] Feature(r,g,pitch,corr) Generator(c++) for pytorch
- [X] DNNModel pytorch
- [X] DNNModel c++ implementation
- [ ] Pretrained model
- [X] Postfiltering (done by [@TeaPoly](https://github.com/TeaPoly ) )


## Requirements
 - CMake
 - Sox
 - Python>=3.6
 - Pytorch

## Prepare sampledata
1. download and sythesize data DNS-Challenge 2020 Dataset before excute utils/run.sh for training. 
```shell
git clone -b interspeech2020/master  https://github.com/microsoft/DNS-Challenge.git
```
2. Follow the Usage instruction in DNS Challenge repo(https://github.com/microsoft/DNS-Challenge) at interspeech2020/master branch. please modify save directories at DNS-Challenge/noisyspeech_synthesizer.cfg sampledata/speech and sampledata/noise each.

## Build & Training
This repository is tested on Ubuntu 20.04(WSL2)

1. setup CMake build environments
```
sudo apt-get install cmake
```
2. make binary directory & build
```
mkdir bin && cd bin
cmake ..
make -j
cd ..
```

3. feature generation for training with sampleData
```
bin/src/percepNet sampledata/speech/speech.pcm sampledata/noise/noise.pcm 4000 test.output
```

4. Convert output binary to h5
```
python3 utils/bin2h5.py test.output training.h5
```

5. Training
run utils/run.sh
```shell
cd utils
./run.sh
```

6. Dump weight from pytorch to c++ header
```
python3 dump_percepnet.py model.pt
```

7. Inference
```
cd bin
cmake ..
make -j1
cd ..
bin/src/percepNet_run test_input.pcm percepnet_output.pcm
```



## Acknowledgements
[@jasdasdf]( https://github.com/jasdasdf ), [@sTarAnna]( https://github.com/sTarAnna ), [@cookcodes]( https://github.com/cookcodes ), [@xyx361100238]( https://github.com/xyx361100238 ), [@zhangyutf]( https://github.com/zhangyutf ), [@TeaPoly](https://github.com/TeaPoly ), [@rameshkunasi]( https://github.com/rameshkunasi ),  [@OscarLiau]( https://github.com/OscarLiau ), [@YangangCao]( https://github.com/YangangCao ), [Jaeyoung Yang]( https://www.linkedin.com/in/jaeyoung-yang-354b21146 )

[IIP Lab. Sogang Univ]( http://iip.sogang.ac.kr/) 



## Reference
https://github.com/wil-j-wil/py_bank

https://github.com/dgaspari/pyrapt

https://github.com/xiph/rnnoise

https://github.com/mozilla/LPCNet
