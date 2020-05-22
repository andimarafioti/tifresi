import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='tifresi',
    version='0.1.4',
    description='Time Frequency Spectrogram Inversion',
    url='https://github.com/andimarafioti/tifresi',
    author='Andrés Marafioti, Nathanael Perraudin, Nicki Hollighaus',
    author_email='nathanael.perraudin@sdsc.ethz.ch',
    license='MIT',
    packages=setuptools.find_packages(),
    zip_safe=False,
    long_description=long_description,
    long_description_content_type="text/markdown",
    extras_require={'testing': ['flake8', 'pytest', 'jupyterlab', 'twine', 'setuptools', 'wheel']},
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Developers', 'Natural Language :: English',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux', 'Programming Language :: C',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering'
    ],
    install_requires=[
        'cython', 'ltfatpy', 'numpy', 'numba', 'librosa', 'matplotlib'
    ],
    python_requires='>=3.6',
)