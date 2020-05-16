from keras import optimizers
from keras import backend as K
from keras import layers
from keras.datasets import mnist
from keras.datasets import cifar10
import numpy as np

def vgg16_layers(inputs,classes=10,dropout_rate=0.2,activation_ch='softmax'):
	# stage 1
	x = layers.Conv2D(filters=64, 
					  kernel_size=(3, 3),
					  activation='relu',
					  padding='same',
					  name='stage1_conv1')(inputs)
	x = layers.Conv2D(filters=64, 
					  kernel_size=(3, 3),
					  activation='relu',
					  padding='same',
					  name='stage1_conv2')(x)
	x = layers.MaxPooling2D(pool_size=(2, 2), 
							strides=(2, 2), 
							name='stage1_pool')(x)

	# stage 2
	x = layers.Conv2D(filters=128, 
					  kernel_size=(3, 3),
					  activation='relu',
					  padding='same',
					  name='stage2_conv1')(x)
	x = layers.Conv2D(filters=128, 
					  kernel_size=(3, 3),
					  activation='relu',
					  padding='same',
					  name='stage2_conv2')(x)
	x = layers.MaxPooling2D(pool_size=(2, 2), 
							strides=(2, 2), 
							name='stage2_pool')(x)

	# stage 3
	x = layers.Conv2D(filters=256, 
					  kernel_size=(3, 3),
					  activation='relu',
					  padding='same',
					  name='stage3_conv1')(x)
	x = layers.Conv2D(filters=256, 
					  kernel_size=(3, 3),
					  activation='relu',
					  padding='same',
					  name='stage3_conv2')(x)
	x = layers.Conv2D(filters=256, 
					  kernel_size=(3, 3),
					  activation='relu',
					  padding='same',
					  name='stage3_conv3')(x)
	x = layers.MaxPooling2D(pool_size=(2, 2), 
							strides=(2, 2), 
							name='stage3_pool')(x)

	# stage 4
	x = layers.Conv2D(filters=512, 
					  kernel_size=(3, 3),
					  activation='relu',
					  padding='same',
					  name='stage4_conv1')(x)
	x = layers.Conv2D(filters=512, 
					  kernel_size=(3, 3),
					  activation='relu',
					  padding='same',
					  name='stage4_conv2')(x)
	x = layers.Conv2D(filters=512, 
					  kernel_size=(3, 3),
					  activation='relu',
					  padding='same',
					  name='stage4_conv3')(x)
	x = layers.MaxPooling2D(pool_size=(2, 2), 
							strides=(2, 2), 
							name='stage4_pool')(x)

	# stage 5
	x = layers.Conv2D(filters=512, 
					  kernel_size=(3, 3),
					  activation='relu',
					  padding='same',
					  name='stage5_conv1')(x)
	x = layers.Conv2D(filters=512, 
					  kernel_size=(3, 3),
					  activation='relu',
					  padding='same',
					  name='stage5_conv2')(x)
	x = layers.Conv2D(filters=512, 
					  kernel_size=(3, 3),
					  activation='relu',
					  padding='same',
					  name='stage5_conv3')(x)
	if x.shape[1] > 1:
		x = layers.MaxPooling2D(pool_size=(2, 2), 
								strides=(2, 2), 
								name='stage5_pool')(x)

	
	x = layers.Flatten(name='flatten')(x)
	x = layers.Dense(units=4096,
					 activation='relu',
					 name='fr1')(x)
#	 x = layers.Dropout(0.2)(x)
	x = layers.Dense(units=4096, 
					 activation='relu', 
					 name='fr2')(x)
#	 x = layers.Dropout(0.2)(x)
	x = layers.Dense(units=1000, 
					 activation='relu', 
					 name='fr3')(x)
	x = layers.Dropout(dropout_rate, 
					   name='dropout')(x)
	x = layers.Dense(classes,
					 activation=activation_ch, 
					 use_bias=True, 
					 kernel_initializer='glorot_uniform', 
					 bias_initializer='zeros',  
					 name='predictions')(x)	
	return x

def select_optimiser(opt,learning_rate):
    if opt=='sgd':
        opt_type =  optimizers.SGD(lr=learning_rate, 
                                         decay=1e-6, 
                                         momentum=0.9, 
                                         nesterov=True)
    if opt=='adagrad':
        opt_type =  optimizers.Adagrad(lr=learning_rate, 
                                             epsilon=1e-08)
    if opt=='adam':
        opt_type = optimizers.Adam(lr=learning_rate, 
                                         beta_1=0.9, 
                                         beta_2=0.999, 
                                         epsilon=1e-08, 
                                         amsgrad=False)
    return opt_type

def evaluate_test(ypred,ytrue):
	true_val = []
	error_val = []
	for i in range(len(ypred)):
		if np.argmax(ypred[i]) == np.argmax(ytrue[i]):
			true_val.append(i)
		else:
			error_val.append(i)
	return true_val, error_val