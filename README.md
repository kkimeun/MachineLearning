# Machine Learning
---

## Introduction
This tutorial has been created based on the talk [Towards the Explainable AI in HEP](https://indico.cern.ch/event/1345869/timetable/).
There are 6 parts introducing from the simplest BDTs to the modified ParticleNet architecture.

### Classifying QCD multijets vs TT-hadronic
0. Plotting input distributions
    - Input distributions used in this classification problem have been shown.
1. Classification using [Random Forests](https://link.springer.com/article/10.1023/A:1010933404324)
    - MDI importance and Permutation importance
    - Mitigating an overfitting problem by using permuattion importance of train and valid set
2. Classification using Deep Neural Networks
    - Simple DNN models using [pytorch](https://pytorch.org/)
    - Attribution methods, applying [Saliency](https://arxiv.org/abs/1711.00867) and [Integrated Gradients](https://arxiv.org/abs/1703.01365) using [CapTum](https://captum.ai/)

### Classifying Light Charged Higgs vs. tt+Z
3. Graph Visualisation
    - Loading dataset and visualising event graphs using [networkx](https://networkx.org/)
4. Modified ParticleNet
    - Introducing [ParticleNet](https://arxiv.org/abs/1902.08570) architecture and its implementatin using [pyg](https://pytorch-geometric.readthedocs.io/en/latest/)
    - Comparing performance with / without giving edge\_attributes
    - Visualising egde indices with their attention masks
5. Hyperparameter Optimization
    - Optimizing one of the hyperparameters - edge\_dropout\_p
    - How to save and reload the pretrained models
6. Surrogated Models
    - How to interpret the input-output relating in model-agnostic approach
    - Training surrogated models

## Environment Setup
There are various way to configure the environment. Here I assume you are using the CMS2 machine to run the notebooks.
I recommend to follow the setting that used in development stage:
- I used conda environment + VS-code with remote access for the development.
- Install [miniconda]() in your local home directory. Activate the conda environment and install the following packages:
```bash
source $CONDA_HOME/bin/activate
conda config --set channel_priority strict
conda create -c conda-forge --name $YOUR_ENVIRONMENT_NAME root python=3.11
conda activate $YOUR_ENVIRONMENT_NAME
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
pip install numpy pandas matplotlib scikit-learn seaborn tensorboard
pip install ipykernel jupyter
```
Because of the file size issue, you should copy the input dataset to your own directory:
```bash
cp -r /data9/Users/choij_public/MachineLearning/DATA $YOUR_TUTORIAL_DIRECTORY/DATA
# e.g.
# cp -r /data9/Users/choij_public/MachineLearning /data9/Users/choij/SNU-project/4.SNU-CMS/Tutorials/MachineLearning/DATA
```

---
- Author: Jin Choi
- Contact: choij@cern.ch
- Last Update: November 27, 2023
