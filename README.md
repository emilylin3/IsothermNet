# IsothermNet
Code release for [paper]

Graph neural networks for material property prediction of MOFs

![Alt text](figs/main.svg)

## Installation

The required Python packages can be found in the Dockerfile. 

## Data

The partial dataset can be found in ```./data```, and the full dataset can be found in the [Zenodo repository](link) under ```full_dataset.zip```. The compressed file contains: 
- ```**X_dataset_electro_xyz_bond_struc.pth**```: post-processed (featurized with ```dataProcessing.py```) structural information
- ```**texturalProperties_vol.xlsx**```: textural properties
- ```**y_dataset19.pth**```: uptake data in g/g
- ```**H_dataset.pth**```: heat of adsorption data in kJ/mol

All MOF samples are sourced from the Quantum MOF (QMOF) database (of the 20,375 MOFs, only 5,394 are CO_{2} adsorption-capable based on the kinetic diameter of a CO_{2} molecule). The crystallographic (.cif) files for each MOF structure can be obtained [here](https://github.com/Andrew-S-Rosen/QMOF/) [1,2].

The input data can be downloaded here (from Zenodo):
```
wget https://zenodo.org/api/files/273e913a-e11d-46e1-96dc-a28497c49d36/data.tar.gz
```

## Training IsothermNet

From configs.py file, 
1. From ```configs.py``` file, load checkpoint, hyperparameter set, and featurized structural inputs (if they exist). If loading the final best model, set ```load_checkpoint = True``` and ```num_epoch = 0```.
   ```
   # Loading checkpoints/data
   load_checkpoint = False        # if False: don't load pre-existing checkpoint, else load best model
   load_hp = True                 # if True: load optimal hyperparameter set, else refine with optuna
   run_dataProcess = False        # if False: don't run featurization on structure, else load featurized set
   ```
2. Run ```train_isothermnet.py``` to train the model and predict on an unseen test set. 

An example of a fully-trained model for the 50 bars case can be found in the [Zenodo repository](link) under ```trained_model_50_bars.zip```. The compressed file contains the checkpoints, best model, and results. 

## Using the Descriptors


## Citing
If you found this work useful, please consider citing: 

## Acknowledgements
This work used the Engaging OnDemand clusters at MIT Office of Research Computing and Data (ORCD). This work additionally used Bridges-2 at Pittsburgh Supercomputing Center (PSC) through allocation MCH230021 from the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program, which is supported by National Science Foundation Grants No. 2138259, 2138286, 2138307, 2137603, and 2138296. This work is also supported by the National Science Foundation Graduate Research Fellowship under Grant No. 2141064. 

## References
[1] A.S. Rosen, S.M. Iyer, D. Ray, Z. Yao, A. Aspuru-Guzik, L. Gagliardi, J.M. Notestein, R.Q. Snurr. "Machine Learning the Quantum-Chemical Properties of Metal–Organic Frameworks for Accelerated Materials Discovery", Matter, 4, 1578-1597 (2021). DOI: 10.1016/j.matt.2021.02.015.
[2] A.S. Rosen, V. Fung, P. Huck, C.T. O'Donnell, M.K. Horton, D.G. Truhlar, K.A. Persson, J.M. Notestein, R.Q. Snurr. "High-Throughput Predictions of Metal–Organic Framework Electronic Properties: Theoretical Challenges, Graph Neural Networks, and Data Exploration," npj Comput. Mat., 8, 112 (2022). DOI: 10.1038/s41524-022-00796-6.
