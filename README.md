# drlse-cuda
CUDA implementation of "Distance Regularized Level Set Evolution and Its Application to Image Segmentation" by Li et al

![input](https://raw.githubusercontent.com/1danielcoelho/drlse-cuda/master/input.png "input sample image") ![output](https://raw.githubusercontent.com/1danielcoelho/drlse-cuda/master/output.png "output zero-level set")

It is capable of running this modality of level set algorithm with iterative calls to a few CUDA kernels. On a NVIDIA GTX 1060 6GB it can segment a 256x256 image in a little under 4s, but there is a ton of room for optimizations.

# Dependencies
* Cuda v9.1 

# Installation
* Easiest way would be to clone the `0_Simple\template` example project from CUDA samples, put the code in there and set it up
* Solution and project files for Visual Studio 2015 are also provided

# Usage
* Just run the program, it will output (or overwrite) `input.dat` and `output.dat` files on the source directory. These are just packed 256*256 32-bit floats. For easy viewing, I recommend using [ImageJ](https://imagej.nih.gov/ij/download.html), and picking Import->Raw, choosing "32-bit Real" for the image type, 256x256 pixels and checking "Little-endian byte order", if that is the case for you
* You can configure the DRLSE parameters within the runCUDA function, but be aware: They are extremely sensitive and interdependent. See [my other repo](https://github.com/1danielcoelho/GA_DRLSE_CUDA) for a program that uses genetic algorithms to find those out for you 

# TODO
* Narrowband optimizations
