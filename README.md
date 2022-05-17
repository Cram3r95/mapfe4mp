conda create --name efficient-goals-motion-prediction python=3.8

pip install 
    prodict
    torch
    pyyaml

Download argoverse-api (1.0) in another folder (out of this directory).
Go to the argoverse-api folder:
    pip install -e . (N.B. You must have the conda environment activated in order to have argoverse as a Python package of your environment)