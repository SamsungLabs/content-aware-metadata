### Official repository for the paper

# Learning sRGB-to-Raw-RGB De-rendering with Content-Aware Metadata @CVPR'22

# [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Nam_Learning_sRGB-to-Raw-RGB_De-Rendering_With_Content-Aware_Metadata_CVPR_2022_paper.pdf) [[arXiv]](https://arxiv.org/abs/2206.01813) [[Supplemental]](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Nam_Learning_sRGB-to-Raw-RGB_De-Rendering_CVPR_2022_supplemental.pdf)

# Usage
- The code runs with Python 3.7 and PyTorch 1.8.
- Install python packages: `torch, torchvision, numpy, scikit-image, opencv-python`
- Download the [dataset](), which is composed of Samsung, Sony, and Olympus cameras.
- Fix the dataset path in each `.sh` file in `./scripts`.
- Run `train_*.sh` for training, and `test_*.sh` for inference.

# Citation
Please cite our paper when you use this code.
```
@InProceedings{Nam_2022_CVPR,
    author    = {Nam, Seonghyeon and Punnappurath, Abhijith and Brubaker, Marcus A. and Brown, Michael S.},
    title     = {Learning sRGB-to-Raw-RGB De-Rendering With Content-Aware Metadata},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {17704-17713}
}
```

# Contact

Please contact [snam0331 AT gmail.com](snam0331@gmail.com) if you have any question about this work.

