# Path-planning-base-on-A3C
## Overview  
This repository contains the implementation of the **A3C-IPP algorithm** described in our paper: *"Path Planning for Adaptive CSI Map Construction with A3C in Dynamic Environments"* and accepted by IEEE TMC. A3C-IPP leverages **Asynchronous Advantage Actor-Critic (A3C)** reinforcement learning, **Gaussian process modeling**, and **mutual information** to optimize the path planning process for constructing CSI fingerprint maps, significantly reducing manual data collection efforts.  

## Abstract  
The fingerprint localization approach using **Channel State Information (CSI)** has become increasingly important with the growing demand for Location-Based Services (LBS). CSI-based methods provide fine-grained information for adequate localization accuracy with low device costs and simple implementation. However, constructing fingerprint maps manually during the offline stage is tedious and time-consuming.  

To address these challenges, A3C-IPP proposes a novel data collection strategy for path planning:  
- **Asynchronous Advantage Actor-Critic (A3C)** is used to transform the optimization problem into a sequential decision-making process.  
- A **multivariate Gaussian process model** and **mutual information** predict the rewards of sampling points to identify the most informative paths.  
- This reinforcement learning-based approach maximizes informative CSI data collection while minimizing manual efforts.  

Extensive experiments in two real-world dynamic environments show that A3C-IPP achieves similar localization accuracy to state-of-the-art algorithms while significantly reducing the CSI collection workload.  

## Features  
- **Reinforcement Learning for Path Planning**: Uses A3C to optimize the data collection process and identify informative paths.  
- **Gaussian Process Modeling**: Predicts sampling rewards using a multivariate Gaussian process.  
- **Mutual Information**: Evaluates the informativeness of sampling points for more efficient CSI collection.  
- **Reduced Manual Labor**: Minimizes tedious data collection tasks while maintaining high localization accuracy.  

## Usage  
The main scripts in this repository include:  
- `LabPathPlanning.py`: Trains the reinforcement learning model for path planning and localization experiments in the Lab Scenario.  
- `CSI_Predict.py`: This is for the Meeting Room Scenario.
- 'LabVIF.py': I make the training process visual as same to 'MeetingVIF.py'.

If you use this work for your research, you may want to cite
```
@article{zhu2021path,
  title={{Path Planning for Adaptive CSI Map Construction with A3C in Dynamic Environments}},
  author={Zhu, Xiaoqiang and Qiu, Tie and Qu, Wenyu and Zhou, Xiaobo and Wang, Yifan and Wu, Oliver},
  journal={IEEE Transactions on Mobile Computing},
  volume={22},
  number={5},
  pages={2925--2937},
  year={2023}
}
```
