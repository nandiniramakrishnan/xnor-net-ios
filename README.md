# xnor-net-ios

# Summary
This project aims to perform image classification using efficient deep learning on iOS device CPUs. This will be done using the binary approximations of popular neural net architectures.

# Background
Due to computational restraints, convolutional neural networks have been restrained to GPUs for obtaining good test results. XNOR-NET aims to bring deep learning to embedded devices by approximating the inputs and weights to the convolutional layers of the networks as binary values. In this scenario, convolutions can be calculated using X-NOR and bit-counting, which can efficiently be done on CPUs.

# References
Rastegari M., Ordonez V., Redmon J., Farhadi A. (2016) XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks. In: Leibe B., Matas J., Sebe N., Welling M. (eds) Computer Vision – ECCV 2016. ECCV 2016. Lecture Notes in Computer Science, vol 9908. Springer, Cham
