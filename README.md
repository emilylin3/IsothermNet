# IsothermNet
Code release for [Neural ordinary differential equations (ODEs) for smooth, high-accuracy isotherm reconstruction, interpolation, and extrapolation](https://www.nature.com/articles/s41524-025-01700-8) (published in *npj Computational Materials*)

![Alt text](figs/main.svg)

IsothermODE is a physically-interpretable model guided by latent dynamics, since it leverages neural ODEs to generate smooth, physically-consistent, full (19-pressure between 0-50 bars) uptake and $\Delta$ H<sub>ads</sub> isotherm predictions. Since most existing models are trained on only low-pressure regimes or generate low-resolution isotherms, they are not suitable for MOF characterization at intermediate or high pressures. Additionally, most adopt per-pressure inferencing, which is inefficient and may yield unsmooth isotherms. The model is similarly adept at predicting tight upper/lower bounds for $\Delta$ H<sub>ads</sub> *via* integrated uncertainty quantification. This addresses two existing limitations in current literature: (1) heat of adsorption ($\Delta$ H<sub>ads</sub>) prediction is still nascent, so the field could benefit from higher-quality inferences, and (2) for the first time, we can model and predict the inherent stochasticity associated with GCMC studies. 

With the understanding that full, 19-point isotherms may be time-consuming and challenging to simulate for future researchers (if they choose to retrain or finetune IsothermODE), we demonstrate the robustness of our model architecture by training with only 5 well-dispersed pressure points (5, 10, 30, and 50 bars), which displayed uptake and $\Delta$ H<sub>ads</sub> prediction qualities that are consistent with the 19-point training. In doing so, we also exhibited very high-performance interpolation and extrapolation capabilities, thus underscoring the generalizability of the model. Overall, even with 5-point training, IsothermODE is superior to current state-of-the-art models in terms of uptake and $\Delta$ H<sub>ads</sub> prediction performance. Moreover, we were able to interpret the learned latent dynamics associated with the model to gain additional insights into structure-property relationships. Finally, we demonstrate IsothermODE’s long-range interpolation/extrapolation capabilities with even more challenging cases (sparse data with large incomplete intervals). In both cases, the model was able to recapitulate the original isotherms. 

To summarize, our work makes the following contributions: 
- Establish a new benchmarking standard for future works (for uptake and $\Delta$ H<sub>ads</sub> predictions)
- Generates tight uncertainty bounds for all $\Delta$ H<sub>ads</sub> predictions to accurately quantify GCMC-based stochasticity
- Demonstrated IsothermODE’s prowess in reconstructing full, high-resolution uptake/$\Delta$ H<sub>ads</sub> isotherms with exceptional interpolation/extrapolation potential, given sparse data (even with large missing intervals)

## Installation

The required Python packages can be found in the Dockerfile. 

## Data

The partial dataset can be found in ```./data/```, and the full dataset can be found in the [Zenodo repository](https://zenodo.org/records/15555513) under ```full_dataset.zip```. The compressed file contains: 
- **```X_dataset_electro_xyz_bond_struc.pth```**: post-processed (featurized with ```dataProcessing.py```) structural information
- **```texturalProperties_vol.xlsx```**: textural properties
- **```y_dataset19.pth```**: uptake data in g/g
- **```H_dataset.pth```**: heat of adsorption ($\Delta$ H<sub>ads</sub>) data in kJ/mol

All MOF samples are sourced from the Quantum MOF (QMOF) database (of the 20,375 MOFs, only 5,394 are CO<sub>2</sub> adsorption-capable based on the kinetic diameter of a CO<sub>2</sub> molecule). The crystallographic (.cif) files for each MOF structure can be obtained [here](https://github.com/Andrew-S-Rosen/QMOF/) [1,2].

The input data can be downloaded here (from Zenodo):
```
wget -O full_dataset.zip "https://zenodo.org/api/records/15555513/files/full_dataset.zip/content"
```

## Training IsothermNet and Predicting Isotherms

1. From ```configs.py``` file, load checkpoint, hyperparameter set, and featurized structural inputs (if they exist). 
   
   ```
   # Loading checkpoints/data
   load_checkpoint = False        # if False: don't load pre-existing checkpoint, else load best model
   load_hp = True                 # if True: load optimal hyperparameter set, else refine with optuna
   run_dataProcess = False        # if False: don't run featurization on structure, else load featurized set
   ```
2. Run ```train_isothermnet.py``` to train the model.
3. Load the best model (set ```load_checkpoint = True``` and ```num_epoch = 0```) and predict on an unseen test set.

Ultimately, IsothermNet can be used to construct full uptake and heat of adsorption ($\Delta$ H<sub>ads</sub>) isotherms.

![Alt text](figs/fig6.svg)

An example of a fully-trained model for the 50 bars case can be found in the [Zenodo repository](https://zenodo.org/records/15555513) under ```trained_model_50_bars.zip```. The compressed file contains the checkpoints, best model, and results. The trained model can be downloaded below:
```
wget -O trained_model_50_bars.zip "https://zenodo.org/api/records/15555513/files/trained_model_50_bars.zip/content"
```

## Using the Descriptors

From the learned adsorption properties, we formulated two sets of universal analytical (A1/A2) and physical (P1/P2) descriptors that effectively bridge MOF structural/surface properties, uptake, and heat of adsorption ($\Delta$ H<sub>ads</sub>) together. The following descriptors are optimized for different pressure regimes and can be summarized as follows: 
- **Analytical descriptors**: ```[A1]``` for high-pressure regime, ```[A2]``` for low-pressure regime (< 10 bars)
- **Physical descriptors**: ```[P1]``` for high-pressure regime, ```[A2]``` for low-pressure regime (< 15 bars)

![Alt text](figs/fig4.png)

To use the descriptors, please refer to ```./descriptors/``` to find the Jupyter demos, which include the parameters for different pressures. The full set of parameters for all descriptors can be found in the [paper](https://www.nature.com/articles/s41524-025-01700-8). 

## Citing

If you found this work useful, please consider citing: 

Lin, E., Zhong, Y., Chen, G. et al. Unified physio-thermodynamic descriptors via learned CO2 adsorption properties in metal-organic frameworks. npj Comput Mater 11, 225 (2025). https://doi.org/10.1038/s41524-025-01700-8

## Acknowledgements

This work used the Engaging OnDemand clusters at MIT Office of Research Computing and Data (ORCD). This work additionally used Bridges-2 at Pittsburgh Supercomputing Center (PSC) through allocation MCH230021 from the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program, which is supported by National Science Foundation Grants No. 2138259, 2138286, 2138307, 2137603, and 2138296. This work is also supported by the National Science Foundation Graduate Research Fellowship under Grant No. 2141064. 

## References

[1] A.S. Rosen, S.M. Iyer, D. Ray, Z. Yao, A. Aspuru-Guzik, L. Gagliardi, J.M. Notestein, R.Q. Snurr. "Machine Learning the Quantum-Chemical Properties of Metal–Organic Frameworks for Accelerated Materials Discovery", Matter, 4, 1578-1597 (2021). DOI: 10.1016/j.matt.2021.02.015.  
[2] A.S. Rosen, V. Fung, P. Huck, C.T. O'Donnell, M.K. Horton, D.G. Truhlar, K.A. Persson, J.M. Notestein, R.Q. Snurr. "High-Throughput Predictions of Metal–Organic Framework Electronic Properties: Theoretical Challenges, Graph Neural Networks, and Data Exploration," npj Comput. Mat., 8, 112 (2022). DOI: 10.1038/s41524-022-00796-6.
