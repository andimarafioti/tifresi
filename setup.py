import setuptools
from setuptools import setup

setup(
    name='tifresi',
    version='0.1.2',
    description='Time Frequency Spectrogram Inversion',
    url='https://github.com/andimarafioti/tifresi',
    author='Andrés Marafioti, Nathanael Perraudin, Nicki Hollighaus',
    author_email='nathanael (dot) perraudin (at) sdsc (dot) ethz (dot) ch',
    license='MIT',
    packages=setuptools.find_packages(),
    zip_safe=False,
    extras_require={'testing': ['flake8', 'pytest', 'jupyterlab']},
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Developers', 'Natural Language :: English',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux', 'Programming Language :: C',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering'
    ],
    install_requires=[
        'cython', 'ltfatpy', 'numpy', 'numba', 'librosa', 'matplotlib'
    ],
)