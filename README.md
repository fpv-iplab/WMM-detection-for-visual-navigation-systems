# World-map misalignment detection for visual navigation systems
Code and documentation of the paper World-map misalignment detection for visual navigation systems

We consider the problem of inferring when the internal map of an indoor navigation system is misaligned with respect to the real world (world-map misalignment), which can lead to misleading directions given to the user. We note that world-map misalignment can be predicted from an RGB image of the environment and the floor segmentation mask obtained from the internal map of the navigation system. Since collecting and labelling large amounts of real data is expensive, we developed a tool to simulate human navigation, which is used to generate automatically labelled synthetic data from 3D models of environments. Thanks to this tool, we gener-
ate a dataset considering 15 different environments, which is complemented by a small set of videos acquired in a real-world scenario and manually labelled for validation purposes. We hence benchmark an approach based on different ResNet18 configurations and compare their results on both synthetic and real images. We
achieved an F1 score of 92.37% in the synthetic domain and 75.42% on the proposed real dataset using our best approach. While the results are promising, we also note that the proposed problem is challenging, due to the domain shift between synthetic and real data, and the difficulty in acquiring real data. The dataset and the
developed tool are publicly available to encourage research on the topic.

# Data genaration tool

The code related to the data generation tool is located in /tool. It includes the project assets and all relevant navigation and randomization scripts. The 3D models of environments (where available) need to be downloaded separately and placed in the /models directory within the Unity project.

# Dataset


# Training and Testing
