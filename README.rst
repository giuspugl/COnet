COnet : Generative Neural Network for CO emission 
===========================================================


This package  provides a suite of simulating CO emission  on  3x3 sq.deg images  (128x128 pixels) extracted from a HEALPIX map, respectively of thermal dust and HI column density. 

For further details see `Puglisi et al. (in prep) <https://giuspugl.github.io/>`_.

Requirements
############

- `tensorflow`
- `astropy`
- `reproject`
- `argparse`
- `Namaster`
- `pynkowski`



Dataset
#######
All the data needed for training have been made available online in the `data <https://drive.google.com/drive/folders/1-Ojcy8tNrLGOMEoMo16fCxARYZM7GLFf?usp=sharing>`_ folder

Usage
#####

Scripts are provided to the user in order to perform:

- Perform Training  `train_cyclegan.py <https://github.com/giuspugl/COnet/blob/main/CycleGAN/train_cyclegan.py>`_,
- Estimate power spectra estimation `runspectra.ipynb <https://github.com/giuspugl/COnet/blob/main/scripts/runspectra.ipynb>`_,
- Minkowski functionals across predictions `runminko.ipynb <https://github.com/giuspugl/COnet/blob/main/scripts/runminko.ipynb>`_,
- Extend  the predictions to full sky emission   `run_fullsky_predictions.ipynb <https://github.com/giuspugl/COnet/blob/main/scripts/run_fullsky_predictions.ipynb>`_, 
 



Pretrained models
#################

`Cycle-GAN  training weights <https://drive.google.com/drive/folders/1-Ojcy8tNrLGOMEoMo16fCxARYZM7GLFf?usp=sharing>`_

Download the model directories  (rename ``checkpoint.txt``  to ``checkpoint`` because google drive automatically add `ext` after download) 


Support
#######

If you encounter any difficulty in installing and using the code or you think you found a bug, please `open an issue
<https://github.com/giuspugl/COnet/issues>`_.
