# AutoCalib

# Overview 
This repository contains the implementation of Zhang's paper on camera calibration. The whole code is implemented from from and calculates the calibration without using opncv's camera calibration function.

## Dependencies
The code is tested with following dependencies and their version:
```
1. python3.5
2. opencv > 4.0.0
3. ubuntu 16.04
4. numpy
5. scipy
```

## How to run the code
```
1. Navigate to the directory where the code is present
2. python3 Wrapper.py --images <path of the images> --output_dir <path to store undistorted images>
```
By default the present working directory is given as the parameters with --images ./Calibration_Imgs and --output_dir ./Output_imgs.
