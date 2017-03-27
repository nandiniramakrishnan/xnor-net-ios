# xnor-net-ios

# Summary
This project aims to perform image classification using efficient deep learning on iOS devices. This will be done using the XNOR-NET architecture and using the ImageNet database.

# Background
Due to computational restraints, convolutional neural networks have been restrained to GPUs for obtaining good test results. XNOR-NET aims to bring deep learning to embedded devices by 
approximating the inputs and weights to the convolutional layers of the networks as binary values. In this scenario, convolutions can be calculated using X-NOR and bit-counting, which can efficiently be done on CPUs.

# Challenges
The XNOR-NET project, available publicly on Github through the Allen Institute of Artificial Intelligence contains a Torch implementation of XNOR-NET. The primary challenge will be to incorporate BNNS functions wherever possible while creating the XNOR-NET model. 

# Goals
The primary goals of this project are to:
Implement XNOR-NET on iOS using the Accelerate framework’s BNNS library
Compare the performance on an iOS device to the original paper’s results

# Schedule
The proposed schedule for this project is as follows:
April 7th - April 14th
Initial research
Develop training and testing framework

April 15th - April 22nd
Translate lua code with Torch implementation into Objective-C

April 22nd - April 29th
Identify optimization areas for utilizing BNNS

April 30th - May 4th
Test, debug and analyze accuracy results

# References
Rastegari M., Ordonez V., Redmon J., Farhadi A. (2016) XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks. In: Leibe B., Matas J., Sebe N., Welling M. (eds) Computer Vision – ECCV 2016. ECCV 2016. Lecture Notes in Computer Science, vol 9908. Springer, Cham
