# muon_tomography
This repository uses CNNs to take in images from muon detectors and output the densities of a 3D object. This code uses muon absorption -- where the muon is absorbed in the object rather than scattered. Hence the images will be 2D images binned in theta and phi with the transmission fraction as the intensity. The transmission fraction is the ratio of the number of muons in a theta-phi bin when the object is present to the number of muons in the theta-phi bin without the object.

This code is inspired by the paper [L. Lim, Z. Qiu, "μ-Net: ConvNext-Based U-Nets for Cosmic Muon Tomography", arXiV:2312.17265 [cs.CV]](https://arxiv.org/abs/2312.17265) and it's [associated repository](https://github.com/jedlimlx/Muon-Tomography-AI).

## Environment
This code was developed using Python 3.10.12. Dependencies:
- Tensorflow (tested with v2.20)
- NumPy (tested with v2.2.6)
- Matplotlib (tested with v3.10.7) 
- Pandas (tested with v2.3.3)

To set up a virtual environment with the necessary packages, you can do the following:
```
python3 -m venv tomography_venv
source ./tomography_venv/bin/activate
pip install -U pip
python3 -m pip install 'tensorflow[and-cuda]'
pip install numpy
pip install matplotlib
pip install pandas
```

## Procedure
At present, this repository uses randomly generated data. For simplicity, we are starting with just one detector. So the model will take in one image and output a 3D tensor of densities.

The inputs to the model training are the following:
- Pandas dataframe of (theta, phi) values when object is present
- Pandas dataframe of (theta, phi) values when object is not present
- 3D array of object densities (this is the training label)

The model will output a 3D array to try to match the 3D object densities.
