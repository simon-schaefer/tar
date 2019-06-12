# TAR

Image reconstruction is a classical problem in computer vision. While most of the state-of-the-art approaches for image reconstruction focus on upscaling only, taking both down- and upscaling into account offers the possibility to improve reconstruction performance while keeping the image compression rate constant. Recent work has shown the feasibility of task aware downscaling for the single-image-super-resolution and the image-colorization task, focussing on the image domain only and excluding external effects.
Within this work advances task aware downscaling is advanced by analysing the effect of perturbation and enhancing robustness and efficiency for super-resolution and colorization in the image domain, by improving both the model architecture and training procedure. As a result for the super-resolution task a model with 44% less parameters could be derived, while having a similar accuracy. For the image-colorization task the reconstruction performance could be improved by 31% with a slightly larger model. In addition, this work introduces a training architecture to extends the task aware downscaling approach from the image to the video domain.

## Installing
The project is implemented in Python3 and based on the pytorch deep learning framework. In order to install these and other requirements as well as setup the project, run:

```
git clone https://github.com/simon-schaefer/tar
cd tar
source scripts/setup_local.bash --download --build --check
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
Implementation inspired by EDSR code (https://github.com/thstkdgus35/EDSR-PyTorch).

[1] Muzhir Al-Ani and Fouad Awad. The jpeg compression algorithm. International Journal of Advances in Engineering and Technology, 2013.
[2] Simon Baker, Daniel Scharstein, J.P. Lewis, Stefan Roth, Michael J. Black, and Richard Szeliski. A database and evaluation methodology for optical flow. IJCV, 2011.
[3] M. Bevilacqua, A. Roumy, C. Guillemot, and M.L Alberi-Morel. Low-complexity single-image super- resolution based on nonnegative neighbor embedding. BMVC, 2012.
[4] Dong C., Loy C.C., He K., and Tang X. Learning a deep convolutional network for image super resolution. ECCV 2014, 2014.
[5] Jose Caballero, Christian Ledig, Andrew P. Aitken, Alejandro Acosta, Johannes Totz, Zehan Wang, and Wenzhe Shi. Real-time video super-resolution with spatio-temporal networks and motion compensation. CoRR, 2016.
[6] W. Dong, L. Zhang, G. Shi, and X. Wu. Image deblurring and super-resolution by adaptive sparse domain selection and adaptive regularization. IEEE Transactions on Image Processing, 2011.
[7] Claude E. Duchon. Lanczos filtering in one and two dimensions. Journal of Applied Meteorology, 1979.
[8] M. Elad. Sparse and redundant representations: From theory to applications in signal and image processing.
Springer Publishing Company, 2010.
[9] Raj Kumar Gupta, Alex Yong-Sang Chia, Deepu Rajan, Ee Sin Ng, and Huang Zhiyong. Image colorization
using similar images. In Proceedings of the 20th ACM International Conference on Multimedia, 2012.
[10] Muhammad Haris, Greg Shakhnarovich, and Norimichi Ukita. Recurrent back-projection network for video
super-resolution. CoRR, 2019.
[11] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition.
CoRR, 2015.
[12] J. Huang, A. Singh, and N. Ahuja. Single image super-resolution from transformed self-exemplars. CVPR,
2015.
[13] Heewon Kim, Myungsub Choi, Bee Lim, and Kyoung Mu Lee. Task-aware image downscaling. In ECCV, 2018.
[14] Jiwon Kim, Jung Kwon Lee, and Kyoung Mu Lee. Accurate image super-resolution using very deep convo- lutional networks. CoRR, 2015.
35
BIBLIOGRAPHY
[15] Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, and Kyoung Mu Lee. Enhanced deep residual net-
works for single image super-resolution. CoRR, 2017.
[16] C. Liu and D. Sun. A bayesian approach to adaptive video super resolution. CVPR, 2011.
[17] D. Liu, Z. Wang, Y. Fan, X. Liu, Z. Wang, S. Chang, and T. Huang. Robust video super-resolution with learned temporal dynamics. In 2017 IEEE International Conference on Computer Vision (ICCV), 2017.
[18] D.R. Martin, C.C. Fowlkes, D. Tal, and J Malik. A database of human segmented natural images and its application to evaluating segmentation algorithms and measuring ecological statistics. ICCV, 2001.
[19] Mehdi S. M. Sajjadi, Raviteja Vemulapalli, and Matthew Brown. Frame-recurrent video super-resolution. CoRR, 2018.
[20] S. Schulter, C. Leistner, and H. Bischof. Fast and accurate image upscaling with super-resolution forests. 2015.
[21] H.Takeda,P.Milanfar,M.Protter,andM.Elad.Super-resolutionwithoutexplicitsubpixelmotionestimation. IEEE Transactions on Image Processing, 2009.
[22] Radu Timofte, Shuhang Gu, Jiqing Wu, Luc Van Gool, Lei Zhang, Ming-Hsuan Yang, Muhammad Haris, et al. Ntire 2018 challenge on single image super-resolution: Methods and results. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, June 2018.
[23] Radu Timofte, Vincent De Smet, and Luc Van Gool. A+: Adjusted anchored neighborhood regression for fast super-resolution. In ACCV, 2014.
[24] Longguang Wang, Yulan Guo, Zaiping Lin, Xinpu Deng, and Wei An. Learning for video super-resolution through HR optical flow estimation. CoRR, 2018.
[25] Xintao Wang, Ke Yu, Shixiang Wu, Jinjin Gu, Yihao Liu, Chao Dong, Chen Change Loy, Yu Qiao, and Xiaoou Tang. ESRGAN: enhanced super-resolution generative adversarial networks. CoRR, 2018.
[26] Yifan Wang, Federico Perazzi, Brian McWilliams, Alexander Sorkine-Hornung, Olga Sorkine-Hornung, and Christopher Schroers. A fully progressive approach to single-image super-resolution. CoRR, 2018.
[27] Zhihao Wang, Jian Chen, and Steven C. H. Hoi. Deep learning for image super-resolution: A survey. CoRR, 2019.
[28] J. Yang, Z. Lin, and S. Cohen. Fast image super-resolution based on in-place example regression. 2013.
[29] Wenming Yang, Xuechen Zhang, Yapeng Tian, Wei Wang, and Jing-Hao Xue. Deep learning for single image
super-resolution: A brief review. CoRR, 2018.
[30] R. Zeyde, M. Elad, and M Protter. On single image scale-up using sparse-representations. Proceedings of the
International Conference on Curves and Surfaces, 2010.
[31] Richard Zhang, Phillip Isola, and Alexei A. Efros. Colorful image colorization. CoRR, 2016.
