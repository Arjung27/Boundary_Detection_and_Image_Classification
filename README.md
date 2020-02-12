# Image_Filters

In this project, a simplified version of probability of boundary detection algorithm was developed, which finds boundaries by examining brightness, color, and texture information across multiple scales (different sizes of objects/image). The output of the algorithm is a per-pixel probability of the boundary detected. The simplified algorithm performs much better when compared to classical edge detection algorithms like Canny and Sobel.

## Running Phase1:
1. From terminal cd into the Code folder of Phase1
2. Run python Wrapper.py to run the code. The code is compatible with both python 2.7 and 3.5. In case, you want to use the already saved clustered maps,
comment the code from line 293-312 and uncomment line 314-316. The program will load the already saved maps, thus generating faster outputs. If the default code is run then it takes around 6 minutes to generate a single pb-lite detection on the image.

## Running Phase2:
For phase two we implemented total 5 deep neural networks. The code for these neural networks is present in Network/Network.py. Due to computational limitations the models were trained for just 20 epochs. The results are not glaringly obvious but we can observe subtle improvements in the results discussed in the report.

In order to run a particular model, one can follow the following steps:
1. Open Train.py, modify the line from Network.Network import <Architecture_name> and then use that architeture name in line 163.
2. Run python Train.py from terminal.

Possible Architecture names are: CIFAR10Model, ResNet, DenseNet, ResNext.
