# IsothermNet
Code release for [paper]

Graph neural networks for material property prediction of MOFs

![Alt text](figs/main.svg)

## Installation & Dataset

Python packages in dockerfile 
partial data in ./data but full data in zenodo[link] (full_dataset.zip)

(replace w/ actual links)
```
wget https://zenodo.org/api/files/273e913a-e11d-46e1-96dc-a28497c49d36/data.tar.gz
wget https://zenodo.org/api/files/273e913a-e11d-46e1-96dc-a28497c49d36/data.tar.gz
```
talk about where your data come from QMOF (link to download cif files)
talk about how data is organized

The crystallographic (.cif) files for each MOF structure can be obtained from the Quantum MOF (QMOF) database (of the 20,375 MOFs, only 5,394 are CO2 adsorption-capable based on the kinetic diameter of a CO_{2} molecule. The full QMOF database can be found [here]([https://pages.github.com/](https://github.com/Andrew-S-Rosen/QMOF/)[1,2]. 

## Training IsothermNet

trained model in zenodo[link] (trained_model_50_bars.zip)

## Using the Descriptors


## Citing


## Acknowledgements
This work used the Engaging OnDemand clusters at MIT Office of Research Computing and Data (ORCD). This work additionally used Bridges-2 at Pittsburgh Supercomputing Center (PSC) through allocation MCH230021 from the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program, which is supported by National Science Foundation Grants No. 2138259, 2138286, 2138307, 2137603, and 2138296. This work is also supported by the National Science Foundation Graduate Research Fellowship under Grant No. 2141064. 

## References
[1] A.S. Rosen, S.M. Iyer, D. Ray, Z. Yao, A. Aspuru-Guzik, L. Gagliardi, J.M. Notestein, R.Q. Snurr. "Machine Learning the Quantum-Chemical Properties of Metal–Organic Frameworks for Accelerated Materials Discovery", Matter, 4, 1578-1597 (2021). DOI: 10.1016/j.matt.2021.02.015.
[2] A.S. Rosen, V. Fung, P. Huck, C.T. O'Donnell, M.K. Horton, D.G. Truhlar, K.A. Persson, J.M. Notestein, R.Q. Snurr. "High-Throughput Predictions of Metal–Organic Framework Electronic Properties: Theoretical Challenges, Graph Neural Networks, and Data Exploration," npj Comput. Mat., 8, 112 (2022). DOI: 10.1038/s41524-022-00796-6.
