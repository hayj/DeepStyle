# DeepStyle

**DeepStyle** provides pretrained models aiming to project text in a stylometric space. The base project consists in a new method of representation learning and a definition of writing style based on distributional properties. This repository contains datasets, pretrained models and other ressources that were used to train and test models.

## Datasets and pretrained model

To get the datasets (i.e. the *R-set* and 22 main *U-sets*), please send me a private message or create a new issue in the repository.

The DeepStyle model (pretrained *DBert-ft*) is available at <https://drive.google.com/file/d/1Y9TMjj04fVhNuJnhzaM4Wmn2CQfZ8r8U/view?usp=share_link>.

## Installation

```bash
git clone https://github.com/hayj/deepstyle
cd deepstyle
python setup.py install
```

Dependencies (tensorflow and transformers) will not be automatically installed since we leave the possibility for users to install newer versions. DeepStyle was tested on tensorflow-gpu `2.0` and transformers `2.4.1`.

## Usage of the *DBert-ft* model

 1. Download the pretrained model available at <https://drive.google.com/file/d/1Y9TMjj04fVhNuJnhzaM4Wmn2CQfZ8r8U/view?usp=share_link> (both the config and weights).
 2. Use the `DeepStyle` class in order to embed documents:

```python
from deepstyle.model import DeepStyle
# Give the folder of the model:
m = DeepStyle("/path/to/the/folder/containing/both/files")
# Sample document:
doc = "Welcome to day two of cold, nasty, wet weather. Ick. Rain is so bad by itself... But when you mix it with a hella cold temperature and nasty wind... Not so much fun anymore."
# Embed a document:
print(m.embed(doc)) # Return a np.ndarray [-0.6553829, 0.3634828, ..., 1.2970213, 0.1685428]
# Get the pretrained model and use its methods (e.g. to get attentions):
m.model # See https://huggingface.co/transformers/model_doc/distilbert.html#tfdistilbertforsequenceclassification
```

In case you have troubles executing this, create a new Python environement and install these package versions:

	pip uninstall -y tensorflow && pip install tensorflow==2.0
	pip uninstall -y transformers && pip install transformers==2.4.1
	pip uninstall -y h5py && pip install h5py==2.10.0

## Experiments

The folder `experiments` contains main experiments of the project. Some parts of the code are notebooks and need to be adapted to your python environment. For long runs, notebooks are converted into python files (e.g. for the *DBert-ft* training).

## "No locks available" issue

In case you get this error, you can set the `HDF5_USE_FILE_LOCKING` env var, for instance, at the beginning of your script:

```python
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
```

## Tested on

	tensorflow-gpu==2.0.0
	transformers==2.4.1
	h5py==2.10.0

With python 3.6 and 3.7, CUDA 10.0 and CUDNN 7.

## Command line demo

Here a command line demo of DeepStyle on Ubuntu 20:

	cd ~/tmp
	mkdir dbert-ft
	cd dbert-ft
	wget http://212.129.44.40/DeepStyle/dbert-ft/config.json
	wget http://212.129.44.40/DeepStyle/dbert-ft/tf_model.h5
	cd ../
	conda create -n dbertft-env -y python=3.7 anaconda
	conda activate dbertft-env
	git clone https://github.com/hayj/deepstyle ; cd deepstyle ; pip uninstall deepstyle -y ; python setup.py install ; cd ../ ; rm -rf deepstyle
	pip install --ignore-installed --upgrade tensorflow==2.0.0
	pip install --ignore-installed --upgrade transformers==2.4.1
	pip install --ignore-installed --upgrade h5py==2.10.0
	ipython -c "from deepstyle.model import DeepStyle ; m = DeepStyle('dbert-ft') ; m.embed('Hello World')"

## Citation

[Link to the publication](https://www.aclweb.org/anthology/2020.wnut-1.30.pdf)

 > Julien Hay, Bich-Liên Doan, Fabrice Popineau, et Ouassim Ait Elhara. Representation learning of writing style. In Proceedings of the 6th Workshop on Noisy User-generated Text (W-NUT 2020), November 2020.

Bibtex format:

	@inproceedings{hay-2020-deepstyle,
	    title = "Representation learning of writing style",
	    author = "Hay, Julien and
	      Doan, Bich-Li\^{e}n and
	      Popineau, Fabrice and
	      Ait Elhara, Ouassim",
	    booktitle = "Proceedings of the 6th Workshop on Noisy User-generated Text (W-NUT 2020)",
	    month = nov,
	    year = "2020"
	}
