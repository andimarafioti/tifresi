# tifresi: Time Frequency Spectrogram Inversion
'tifresi' to be pronounced 'tifreeezy' provide a simple implementation of TF and spectrogam suitable for inversion, i.e. with a high quality phase recovery.
The phase recovery algorithm used is PGHI (phase gradient heap integration).

TODO: Put references ...

## Installation

This repository use the ltfatpy packages that requires a few libraries to be installed. 

1. Be sure that cmake is installed
   * On debian based unix system:
    ```
    sudo apt-get install cmake
    ```
   * On MacOS X using homebrew:
    ```
    brew install cmake
    ```
   * On MacOS X using port:
    ```
    sudo port install cmake
    ```
2. Install `fftw3` and `lapack`
   * On debian based unix system:
    ```
    sudo apt-get install libfftw3-dev liblapack-dev
    ```
   * On MacOS X using homebrew:
    ```
    brew install fftw lapack
    ```
   * On MacOS X using port:
    ```
    sudo port install fftw-3 fftw-3-single lapack
    ```
3. Install cython (required for installing ltfatpy):
    ```
    pip install -r cython
    ```      
4. Install the requirements (You probably wants to create a virtual environment first)
    ```
    pip install -r requirements
    ```    

## Starting
After installation of the requirements, you can check the following notebooks:
* `demo.ipynb` illustrates how to construct a spectrogram and invert it.
* `demo-mel.ipynb` illustrates how to compute a mel spectrogram with the setting used in this repository.




#### Main files
* `utils.py`: Utility to load, preprocess, downsample the signal
* `stft.py`: core objects 
    1. TFGauss: to compute and invert the STFT with a full Gaussian window
    2. GaussTruncTF: to compute and invert the STFT with a truncated Gaussian window  (faster)
* `metrics.py`: compute metrics to evaluate phase and spectrogram quality
* `transforms.py`: useful simple transform function for spectrograms
* `hparams.py`: default parameters
* `phase`: folder containing source code for phase reconstruction using pghi
    - TODO : create the folder and move files
* `tests`: folder containing all tests

    


#### To be done 
* Create a package



