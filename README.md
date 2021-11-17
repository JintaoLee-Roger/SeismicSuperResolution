# SeismicSuperResolution

This is a repository for the paper "Deep Learning for Simultaneous Seismic Image Super-Resolution and Denoising" (IEEE Transactions on Geoscience and Remote Sensing).

The frame and some code are from [sanghyun-son/EDSR-PyTorch](https://github.com/sanghyun-son/EDSR-PyTorch).
[src/loss/msssim.py](src/loss/msssim.py) was modified based on [jorge-pessoa/pytorch-msssim](https://github.com/jorge-pessoa/pytorch-msssim).

## Usage

```
SeismicSuperResolution/
      ├───── data/
      │        ├───── sx/     # high resolution data
      │        ├───── nx2/    # low resolution data
      │        └───── feild/  # feild data
      │                 ├───── kumano2_608x400.dat
      │                 ├───── lulia_592x400.dat
      │                 ├───── tp_352x240.dat
      │                 └───── ...
      ├───── experiment/
      │        ├───── alpha6/
      │        │        ├───── model/
      │        │        │         ├───── model_best.pt # in google drive
      │        │        │         └───── ...
      │        │        └───── ...
      │        └───── ...
      │ 
      └───── src/        
               └───── ...
```

### Dataset

All the data used in this paper is avaliable in google drive [https://drive.google.com/drive/folders/1DuMdclOdeXDgGBOhsHSlEdTB_LvhIH-X?usp=sharing](https://drive.google.com/drive/folders/1DuMdclOdeXDgGBOhsHSlEdTB_LvhIH-X?usp=sharing). And the model `experiment/alpha6/model/model_best.pt` can also be obtained by above google drive link.


### Code 

All code is in the directory `src`.


### Dependencies
- python 3.6.9
- pytorch 1.6.0
- numpy 1.17.4
- cudatoolkit 10.1.243
- matplotlib 3.1.1

## Citation
If you find this work useful in your research, please consider citing:

Plain Text
```
J. Li, X. Wu and Z. Hu, "Deep Learning for Simultaneous Seismic Image Super-Resolution and Denoising," in IEEE Transactions on Geoscience and Remote Sensing, doi: 10.1109/TGRS.2021.3057857.
```

BibTex
```latex
@ARTICLE{deep2021li,
   author={J. {Li} and X. {Wu} and Z. {Hu}},
   journal={IEEE Transactions on Geoscience and Remote Sensing}, 
   title={Deep Learning for Simultaneous Seismic Image Super-Resolution and Denoising}, 
   year={2021},
   volume={},
   number={},
   pages={1-11},
   doi={10.1109/TGRS.2021.3057857}}
```
