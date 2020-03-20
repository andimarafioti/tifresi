# tifresi: Time Frequency Spectrogram Inversion
'tifresi' to be pronounced 'tifreeezy' provide a simple implementation of TF and spectrogam suitable for inversion, i.e. with a high quality phase recovery.
The phase recovery algorithm used is PGHI (phase gradient heap integration).

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
    pip install cython
    ```      
4. Install the requirements (You probably wants to create a virtual environment first)
    ```
    pip install -r requirements.txt
    ```    
5. Clone this repository and install it 
    ```
    git clone https://github.com/andimarafioti/tifresi
    cd tifresi
    pip install .
    ```       

## Starting
After installation of the requirements, you can check the following notebooks:
* `demo.ipynb` illustrates how to construct a spectrogram and invert it.
* `demo-mel.ipynb` illustrates how to compute a mel spectrogram with the setting used in this repository.


## License & citation

The content of this repository is released under the terms of the [MIT license](LICENCE.txt).
Please consider citing our papers if you use it.

```
@inproceedings{marafioti2019adversarial,
  title={Adversarial Generation of Time-Frequency Features with application in audio synthesis},
  author={Marafioti, Andr{\'e}s and Perraudin, Nathana{\"e}l and Holighaus, Nicki and Majdak, Piotr},
  booktitle={International Conference on Machine Learning},
  pages={4352--4362},
  year={2019}
}
```

```
@article{pruuvsa2017noniterative,
  title={A noniterative method for reconstruction of phase from STFT magnitude},
  author={Pr{\uu}{\v{s}}a, Zden{\v{e}}k and Balazs, Peter and S{\o}ndergaard, Peter Lempel},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume={25},
  number={5},
  pages={1154--1164},
  year={2017},
  publisher={IEEE}
}
```

## Developing
As a developer, you can test the package using `pytest`:
```
pip install pytest
```
Then run tests using
```
pytest tifresi
```
You can also use the source code checker `flake8`:
```
pip install flake8
```
Then run tests using
```
flake8 .
```

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


#### TODO
* Add links to papers
* Put the package on pypi
* Improve doc
* Put the documentation on readthedoc or somthing similar



