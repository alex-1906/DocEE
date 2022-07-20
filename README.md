# DocEE - Document-level Event Extraction 

# Table of Contents
1. [Data](#data)
2. [Setup Dependencies](#dependencies)
3. [Reproducing Results (+ link to trained models)](#reproducing-results)
4. [Training](#training)


# Data
The WikiEvents task data can be found [here](https://github.com/raspberryice/gen-arg).
To use it with the code in this repo, place the training, dev, test (json-)files into the corresponding folders in the data directory.

# Setup
**Dependencies:**
- numpy (1.23.1)
- pandas (1.4.3)
- torch (1.12.0)
- transformers (4.20.1)
- tqdm (4.64.0)
- wandb (0.12.21)
- nltk (3.7)

Setup a virtual environment and install necessary dependencies by running the following lines of code:
```
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
**Preprocessing:**  
Make sure to place the raw train, dev, test (jsonl-) files in the corresponding folders. 
For creation of the Ontology files containing feasible roles, relation types and mention types, run preprocessing with --ont_files=True 

```
python preprocess.py --ont_files=True
```

# Training
Example of training a model without soft_mention and at_inference:
```
python train.py --learning_rate=1e-5 --num_epochs=5 --soft_mention=False --at_inference=False 
```

Example of training a model with soft_mention and at_inference:
```
python train.py --learning_rate=1e-6 --num_epochs=5 --soft_mention=True --at_inference=True 
```
