
# DeepStyle

**DeepStyle** provides pretrained models aiming to project text in a stylometric space. The base project consists in a new method of representation learning and a definition of writing style based on distributional properties. This repository contains datasets, pretrained models and other ressources that were used to train and test models.

## Datasets and pretrained model

All datasets (the *R-set* and 22 main *U-sets*) as well as the pretrained *DBert-ft* model are available at <http://212.129.44.40/DeepStyle/>. Other *U-sets* are comming soon.

## Installation

```bash
git clone https://github.com/anonym2020/deepstyle
cd deepstyle
python setup.py install
```

Dependencies (tensorflow and transformers) will not be automatically installed since we leave the possibility for users to install newer versions. DeepStyle was tested on tensorflow-gpu `2.0` and transformers `2.4.1`.

## Usage of the *DBert-ft* model

 1. Download the pretrained model available at <http://212.129.44.40/DeepStyle/dbert-ft/> (both the config and weights).
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

## Experiments

The folder `experiments` contains main experiments of the project. Some parts of the code are notebooks and need to be adapted to your python environment. For long runs, notebooks are converted into python files (e.g. for the *DBert-ft* training).

## "No locks available" issue

In case you get this error, pls set the `HDF5_USE_FILE_LOCKING` env var, for instance, at the beginning of your script:

```python
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
```

## Tested on

	tensorflow-gpu==2.0.0
	transformers==2.4.1

With python 3.6, CUDA 10.0 and CUDNN 7.
