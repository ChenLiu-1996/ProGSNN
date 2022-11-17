# ProGSNN
### Krishnaswamy Lab, Yale University

Contributors:
- Egbert Castro (egbert.castro@yale.edu)
- Dhananjay Bhaskar (dhananjay_bhaskar@brown.edu)
- Chen Liu (chen.liu.cl2482@yale.edu)

## Cite
```
ProGSNN: Deep Multi-Scale Protein Representation Learning using Geometric Scattering
Egbert Castro, Dhananjay Bhaskar, Jackson Grady, Alex Grigas, Michael Perlmutter, Corey S. O'Hern, Smita Krishnaswamy
NeurIPS 2021 Learning Meaningful Representations of Life Workshop
```

## Dependencies
We developed the codebase in a miniconda environment.
Tested on Python 3.9.13 + PyTorch 1.12.1.
How we created the conda environment:
```
conda create --name $OUR_CONDA_ENV pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda activate $OUR_CONDA_ENV
conda install pytorch_geometric torch-scatter pytorch-lightning -c conda-forge
python -m pip install pysmiles graphein phate
conda install pytorch3d -c pytorch3d
conda install scikit-image pillow -c anaconda
```

## Usage
```
cd src/scripts
python run_all.py
```
