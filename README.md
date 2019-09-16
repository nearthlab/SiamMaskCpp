# SiamMaskCpp
* C++ Implementation of [SiamMask](https://github.com/foolwood/SiamMask)
* Efficient network output post-processing using [OpenCV's GPU matrix operations](https://docs.opencv.org/2.4/modules/gpu/doc/per_element_operations.html) instead of numpy
* Faster than original implementation (speed increased from 22fps to 40fps when tested with a single NVIDIA GeForce GTX 1070)

# Convert a SiamMask model to Torch scripts
You can use the models (with the refine module) trained with the original repository [foolwood/SiamMask](https://github.com/foolwood/SiamMask) for inference in C++. Just Follow the instruction in [jiwoong-choi/SiamMask](https://github.com/jiwoong-choi/SiamMask#converting-siammask-model-with-the-refine-module-to-torch-scripts) to convert your own models to Torch script files.

# Download pretrained Torch scripts
Or you can download pretrained Torch scripts from the [release page](https://github.com/nearthlab/SiamMaskCpp/releases). 
These are converted from the pretrained models ([SiamMask_DAVIS.pth](http://www.robots.ox.ac.uk/~qwang/SiamMask_DAVIS.pth) and [SiamMask_VOT.pth](http://www.robots.ox.ac.uk/~qwang/SiamMask_VOT.pth)) in the original repository.

# How to build demo
```bash
git clone --recurse-submodules https://github.com/nearthlab/SiamMaskCpp
cd SiamMaskCpp
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/python3.x/site-packages/torch ..
make
```

# How to run demo
```bash
cd SiamMaskCpp/build
./demo -c ../config_davis.cfg -m ../SiamMask_DAVIS ../tennis
```
