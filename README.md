# Exploring Map-based Features for Efficient Attention-based Vehicle Vehicle Motion Prediction

<img src="data/datasets/argoverse/motion-forecasting/train/goal_points_test/3_step_7_multimodal.png"/>

## Overview
Motion prediction (MP) of multiple agents is a crucial task in arbitrarily complex environments, from social robots to self-driving cars. Current approaches tackle this problem using end-to-end networks, where the input data is usually a rendered top-view of the scene and the past trajectories of all the agents; leveraging this information is a must to obtain optimal performance. In that sense, a reliable Autonomous Driving (AD) system must produce reasonable predictions on time, however, despite many of these approaches use simple ConvNets and LSTMs, models might not be efficient enough for real-time applications when using both sources of information (map and trajectory history). Moreover, the performance of these models highly depends on the amount of training data, which can be expensive (particularly the annotated HD maps). In this work, we explore how to achieve competitive performance on the Argoverse 1.0 Benchmark using efficient attention-based models, which take as input the past trajectories and map-based features from minimal map information to ensure efficient and reliable MP. These features represent interpretable information as the driveable area and plausible goal points, in opposition to black-box CNN-based
methods for map processing. our code is publicly available at ([Code](https://github.com/Cram3r95/mapfe4mp)).

<!-- Second, the system is validated ([Qualitative Results](https://cutt.ly/uk9ziaq)) in the CARLA simulator fulfilling the requirements of the Euro-NCAP evaluation for Unexpected Vulnerable Road Users (VRU), where a pedestrian suddenly jumps into the road and the vehicle has to avoid collision or reduce the impact velocity as much as possible. Finally, a comparison between our HD map based perception strategy and our previous work with rectangular based approach is carried out, demonstrating how incorporating enriched topological map information increases the reliability of the Autonomous Driving (AD) stack. Code is publicly available ([Code](https://github.com/Cram3r95/map-filtered-mot)) as a ROS package. -->

## Requirements

<!-- Note that due to ROS1 limitations (till Noetic version), specially in terms of TF ROS package, we limited the Python version to 2.7. Future works will integrate the code using ROS1 Noetic or ROS2, improving the version to Python3. -->

<!-- - Python3.8 
- Numpy
- ROS melodic
- HD map information (Monitorized lanes)
- scikit-image==0.17.2
- lap==0.4.0 -->
- OpenCV==4.1.1
- YAML
- ProDict
- torch (1.8.0+cu111)
- torchfile (0.1.0)
- torchsummary (1.5.1)
- torchtext (0.5.0)
- torchvision (0.9.0+cu111)

## Get Started and Usage
Coming soon ...
## Quantitative results
Coming soon ...
## Qualitative results
Coming soon ...

  - TO DOs:

	- [ ] Study Adaptive Average Pool 2D to apply in the LSTM based encoder (at this moment we are taking final_h =    states[0], so the last one, instead of average, max pool, etc.) and the linear feature of the physical_attention in order to receive different width x height images and get a fixed-size output 
    - [ ] Study the attention module, different approaches, specially the Social Attention module

conda create --name efficient-goals-motion-prediction python=3.8 \
conda install -n carlos_efficient-goals-motion-prediction ipykernel --update-deps --force-reinstall

python3 -m pip install --upgrade pip \
python3 -m pip install --upgrade Pillow \

pip install \
    prodict \
    torch \
    pyyaml \
    torchvision \
    tensorboard \
    torchstat

Download argoverse-api (1.0) in another folder (out of this directory). \
Go to the argoverse-api folder: 
```
    pip install -e . (N.B. You must have the conda environment activated in order to have argoverse as a Python package of your environment)
```