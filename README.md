# AdaptiveScattering
Code for the paper: Changhong Wang, Emmanouil Benetos, Vincent Lostanlen and Elaine Chew. [Adaptive Scattering Transforms for Playing Technique Recognition](https://ieeexplore.ieee.org/document/9729446), IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP), 2022.

## How to run
### Get code
`git clone https://github.com/changhongw/AdaptiveScattering.git`

### Data
Download the CBFdataset from https://zenodo.org/record/5744336.

### Adaptive scattering feature extraction
`python feature_extraction.py`

The adaptive scattering feature extraction code is based on the [ScatNet](https://www.di.ens.fr/data/software/scatnet/), a MATLAB implementation of the scattering transform. However, we can run the code in Python using [subprocess](https://docs.python.org/3/library/subprocess.html) module. 

### Playing Technique Recognition
`python playing_technique_recognition.py > output.txt`

## Citation
```
@article{wang2022adaptive,
  title={Adaptive scattering transforms for playing technique recognition},
  author={Wang, Changhong and Benetos, Emmanouil and Lostanlen, Vincent and Chew, Elaine},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume={30},
  pages={1407--1421},
  year={2022},
  publisher={IEEE}
}
```
