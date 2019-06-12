# TAR

Image reconstruction is a classical problem in computer vision. While most of the state-of-the-art approaches for image reconstruction focus on upscaling only, taking both down- and upscaling into account offers the possibility to improve reconstruction performance while keeping the image compression rate constant. Recent work has shown the feasibility of task aware downscaling for the single-image-super-resolution and the image-colorization task, focussing on the image domain only and excluding external effects.
Within this work advances task aware downscaling is advanced by analysing the effect of perturbation and enhancing robustness and efficiency for super-resolution and colorization in the image domain, by improving both the model architecture and training procedure. As a result for the super-resolution task a model with 44% less parameters could be derived, while having a similar accuracy. For the image-colorization task the reconstruction performance could be improved by 31% with a slightly larger model. In addition, this work introduces a training architecture to extends the task aware downscaling approach from the image to the video domain.

## Installing
The project is implemented in Python3 and based on the pytorch deep learning framework. In order to install these and other requirements as well as setup the project, run:

```
git clone https://github.com/simon-schaefer/tar
cd tar
source scripts/setup_local.bash --build --download --check
```

## Training
To train a model the main function has to be called which automatically starts training using the default parameters. Thereby, a wide range of parameters is available (defined in src/tar/inputs.py), determining the training hyperparameters, the datasets used, logging information and a set of task dependent parameters. A lot of training scenarios are thereby predefined as templates such as ISCALE_AETAD_LARGE_DIV2K_4 trains the AETAD_LARGE model on the DIV2K training dataset for the single-image super-resolution problem with scaling factor 4.

```
python3 main.py
```
For evaluation the --valid_only flag can be used, for model loading a model the --load flag.

## Built With

* [PyTorch](https://pytorch.org) - PyTorch deep learning framework

## References
Heewon Kim, Myungsub Choi, Bee Lim, and Kyoung Mu Lee. Task-aware image downscaling. In ECCV, 2018.
