# NL-Vocoder Vocoder

Synthesized samples are available at https://nlvocoder.github.io/nlvocoder.html <br /><br />
&nbsp;
&nbsp;
<br />
This repository contains code for the implementation of the Non-linear vocoder.

Follow the steps below to install the dependenies, train the non-linear vocoder, and synthesize from the trained vocoder.

## Install dependencies with virtual environment using requirement.yml file.

All the dependencies are placed in the requirement.yml file.
Create a virtual environment using the requirement file.
```
conda env create -f requirement.yml
```

Activate the virtual environment.
```
conda activate nl_vocoder       #nl_vocoder is the name of the virtual environment.
```

## Data Preparation
Copy the wavfiles to the data folder.

Divide the total wavfiles into **wavs** and **wav_test** folders for train and test respectively.
```
python prep_data.py
```

## Training
```
CUDA_VISIBLE_DEVICES=0 python train.py
```

The trained model is stored in the **weights** folder.


## Synthesis

Test data is in the **data/wav_test** folder.
```
python analysis_synthesis.py --checkpoint weights/weights-0278.h5 --input data/wav_test/S0001_0018.wav
```

The synthesized samples are available at **synthesized** folder.


