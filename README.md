# MNIST_Neural_Nets


#Vanilla_NN
  Vanilla_NN is my first working prototype of a neural net. In order to train this network, input the training data, X, 
and the training labels, Y, into sgd() with the desired batch size, bs, which I usually set to 20. Then you can test the
network by inputing the the test images, Xt, and the test labels, Yt, into test(). This will return the networks accuracy. 
  Vanilla_NN uses batches in its gradient descent algorithim but not randomized batches because the data imported from keras 
is already randomized. In order to save proccesing time, I have been running this on my laptop, I did not want to waste
resources in order to rerandomize the data set.
