# vgg16
VGG16 implementation on MNIST and CIFAR10

Paper: https://arxiv.org/abs/1409.1556


To run VGG16 with MNIST dataset
python train.py --dataset mnist --model vgg16 --reshape '(32,32)' --batch_size 128 --epoch 10 --learning_rate 0.01 --dropout_rate 0.2 --activation_ch softmax --optimizer_ch sgd



To run VGG16 with CIFAR10 dataset
python train.py --dataset cifar --model vgg16 --reshape '(32,32)' --batch_size 128 --epoch 10 --learning_rate 0.01 --dropout_rate 0.2 --activation_ch softmax --optimizer_ch sgd
