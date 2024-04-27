# Implementation of SPD-MEF
This is a non-official Qt C++ implementation of the algorithm SPD-MEF in paper "Robust Multi-Exposure Image Fusion: A Structural Patch Decomposition Approach".

## Attention
This implementation can only process GRAY scale images. This means if you input a series of 3-channel images, what you get is only a fused GRAY(1-channel) image.

## Usage 
1. Build Environment 
    This project is based on QT c++ and opencv3.4.16 and boost1.65.1, so make sure you have built this environment first.
2. Generate the Fused Image 
    In `main.cpp`, change the path to your own image folder. Pay attention: the path must be end with '\\'. Make sure your input is a folder.

## Details

### Funcitions
1. DoLinearAddAndMean(): N images each pixel add respectively then divide by N
2. DoMertensMEF(): exposure fusion implementation
3. DoSPDMEF(): SPD-MEF implementation
4. DoGaussMEF(): N images fusion by calculating each pixel gauss weight
5. GetImagesMat(): read images
6. GetImagesGrayMat(): convert color images to gray images, maybe you should change this
7. AdjustImageSize(): addjust images size
8. MatchHistograms(): c++ implementation of matchHistograms function in skimage

### Parameters
1. image_max_worh_size_:set the max size of width or height
2. patch_size_: SPDMEF patch size
3. step_size_: SPDMEF step 
4. Ts_: SPDMEF Ts
5. Tm_: SPDMEF Tm
6. global_gaussian_: SPDMEF σg
7. local_gaussian_: SPDMEF σl
8. float p_: SPDMEF p







