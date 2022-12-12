# Distributionally Robust Multi-Task Optimization for Fair Skin Cancer Classification

> Codebase for my *CS 221: Artificial Intelligence* project. 

**Context**: my project revolves around designing a new multi-task optimization paradigm, adapting the technique of group distributionally robust optimization, for unbiased skin cancer classification given data imbalances across groups. The motivation is to reduce skin color bias in machine learning for this problem given imbalanced data. 

<p align='center'>
    <img alt="picture 1" src="https://cdn.jsdelivr.net/gh/minimatest/vscode-images@main/images/6c145b7f5ab74d9eda99d36c183e18351ff0aa2034d6a633f91bae5e3baa83a7.png" width="50%" />  
</p>


## File structure 


- `data`: contains the raw data and processing scripts to build the task distributions. 
- `multitask`: codebase for implementing the main, novel approach of this project: multi-task group DRO. 
- `supervised`: codebase for implementing traditional supervised learning to be used as a baseline comparison. 
- `group-dro`: codebase for implementing traditional group DRO (single task). Used for comparison purposes. 
- `logs`: training and test logs. 


## Methods 


Our goal is to classify skin lesions as cancerous or not from images. We use the [SynthDerm](https://affect.media.mit.edu/dissect/synthderm/) synthetic skin lesion image dataset which contains an abundance of positive and negative skin lesion synthetic images for all six of the [Fitzpatrick skin color types](https://en.wikipedia.org/wiki/Fitzpatrick_scale). 

The setup in this multi-task learning problem is to treat each fitzpatrick type as a task $T_i$ where each task encodes the respective distribution: 

$$
\mathscr{T}_i \triangleq\left\{p_i(\mathbf{x}), p_i(\mathbf{y} \mid \mathbf{x}), \mathscr{L}_i\right\}.
$$

The important part is that for skin types 1, 2, and 3, we use 50 training samples but for skin types 4, 5, and 6, we use only 25 training samples (data imbalance on a per-task basis). This to reflect the fact that most skin cancer datasets that exist in the real world often have group imbalances in the data, favoring the lighter skin colors over the darker skin colors which creates downstream bias upon performing inference.  

In the multi-task setup, our goal is to learn a model that is capable of inference for each task all at once (i.e., to make a prediction, we'd input a sample from each of the six tasks and we'd retrieve the predictions for all 6 at once). In traditional multi-task learning, the objective is to minimize the sum of losses across all tasks:  

$$
\min _\theta \sum_{i=1}^T \mathscr{L}_i\left(\theta, \mathscr{D}_i\right).
$$

However, our novel, **proposed approach is to minimize the worst-case loss across all tasks** to ensure robustness across all tasks: 

$$
\min _\theta  \max _{T_i} (\mathscr{L}_i\left(\theta, \mathscr{D}_i\right)).
$$


The hypothesis is that if our objective is to maximize the performance of the worst group/task/distribution in the multi-task setup, then the model will achieve less biased results. 

## Results 

| Method   | Skin Type's {1, 2, 3} Accuracy | Skin Type's {4, 5, 6} Accuracy | Logs |
|---------|---|---| --- |
| Standard Supervised Learning       | 80% | 73% | [logs/supervised-learning](logs/supervised-learning) |
| Single-Task Group DRO       | 63% | 53% | [logs/single-task-group-dro](logs/single-task-group-dro) |
| Standard Multi-Task Learning     | 73% | 66% | [logs/standard-multi-task-learning](logs/standard-multi-task-learning)
| **Multi-Task Group DRO** (our approach)     | 60% | 66% | [logs/multi-task-group-dro](logs/multi-task-group-dro)

Our experiments show that our approach (Multi-Task Group DRO) achieves the best results for the imbalanced skin types (4, 5, and 6) relative to the results for skin types (1, 2, and 3, which were the skin types that had more training data). However, traditional supervised learning beats out all methods for overall accuracy. 
