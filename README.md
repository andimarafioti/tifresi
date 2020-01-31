# stft4pghi
STFT transforms suitable for use with PGHI (phase gradient heap integration)

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
* `src`: folder containing source code for phase reconstruction using pghi
    - TODO : create the folder and move files
* `tests`: folder containing all tests

    


#### To be done 
* Create a package

* `preprocess.py`: contain the main function for the signal (maybe something like this, also maybe to be merged with utils.py)
    ```
    def make_spectrograms(y, a, M, n_mels, sr=22050):
        '''Parse the wave file in `fpath` and
        Returns normalized melspectrogram and linear spectrogram.
        Args:
          y  : sound file
        Returns:
          mel: A 2d array of shape (T, n_mels) and dtype of float32.
          mag: A 2d array of shape (T, 1+n_fft/2) and dtype of float32.
        '''

        tfsystem = GaussTF(a=a, M=M)
        linear = tfsystem.dgt(y)

        # magnitude spectrogram
        mag = np.abs(linear)  # (1+n_fft//2, T)
        mag = mag/np.max(mag)

        # mel spectrogram
        mel_basis = librosa.filters.mel(sr, M, n_mels)  # (n_mels, 1+n_fft//2)
        mel = np.dot(mel_basis, mag)  # (n_mels, t)

        # to decibel
        mel = np.log10(np.maximum(1e-5, mel))/2.5+1
        mag = np.log10(np.maximum(1e-5, mag))/2.5+1
        assert(np.max(mag)<=1)
        assert(np.min(mag)>=-1)    

        # Transpose
        mel = mel.astype(np.float32)  # (T, n_mels)
        mag = mag.astype(np.float32)  # (T, 1+n_fft//2)

        return mel, mag

    ```

