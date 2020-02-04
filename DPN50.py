"""DPN50 model for tensorflow-keras.

Reference paper:

[Dual Path Networks]
(https://arxiv.org/pdf/1707.01629.pdf)

Model adapted from:
- [Tensorflow]
(https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet50.py)
"""
import os
import warnings
import tensorflow as tf
from tensorflow.keras import layers, activations, Input, Model
import tensorflow.keras.backend as backend


def identity_block(input_tensor, kernel_size, filters, stage, block, width_increment):
    """The identity block is the block that has no conv layer at shortcut.
    This block has grouped convolution implementation.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    grouped_convolution = []
    for i in range(CARDINALITY):
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a' + f'_{i}')(input_tensor)
        x = activations.relu(x)
        x = layers.Conv2D(int(round(filters1/CARDINALITY))*2, (1, 1),
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2a' + f'_{i}')(x)
        
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b' + f'_{i}')(x)
        x = activations.relu(x)
        x = layers.Conv2D(int(round(filters1/CARDINALITY))*2, kernel_size,
                          padding='same',
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2b' + f'_{i}')(x)
        
        grouped_convolution.append(x)
    x = layers.concatenate(grouped_convolution, axis=-1)
    
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    x = activations.relu(x)
    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    
    # Split Input tensor to its residual and dense components
    residual_input = layers.Lambda(lambda x: x[:, :, :, :filters3])(input_tensor)
    dense_input = layers.Lambda(lambda x: x[:, :, :, filters3:])(input_tensor)
    # Extract the grwoth increment from the last convolution
    dense_increment = layers.Lambda(lambda x: x[:, :, :, :width_increment])(x)
    
    x = layers.add([residual_input, x])
    x = layers.concatenate([x, dense_input, dense_increment], axis=bn_axis)
    return x

def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               width_increment,
               strides=(2, 2)):
    """A block that has a conv layer at shortcut. This block has grouped
    convolution implementation.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
    # Returns
        Output tensor for the block.
    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    grouped_convolution = []
    for i in range(CARDINALITY):
        if strides == (1, 1):
            x = input_tensor
            # First activation of the network is done before splitting the paths.
            # Checking for strides saves us from introducing a new variable.
            # (https://gsy00517.github.io/deep-learning20200113174731/1603.05027.pdf)
        else:
            x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a' + f'_{i}')(input_tensor)
            x = activations.relu(x)
        x = layers.Conv2D(int(round(filters1/CARDINALITY))*2, (1, 1), strides=strides,
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2a' + f'_{i}')(x)
        
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b' + f'_{i}')(x)
        x = activations.relu(x)
        x = layers.Conv2D(int(round(filters1/CARDINALITY))*2, kernel_size, padding='same',
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2b' + f'_{i}')(x)
        
        grouped_convolution.append(x)
    x = layers.concatenate(grouped_convolution, axis=-1)
    
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    x = activations.relu(x)
    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    
    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    dense_increment = layers.Lambda(lambda x: x[:, :, :, :width_increment])(x)
    
    x = layers.add([shortcut, x])
    x = layers.concatenate([x, dense_increment], axis=bn_axis)
    return x


def DPN50(include_top=True,
             weights=None,
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             **kwargs):
    """Code Adapted from https://github.com/keras-team/keras-applications/
    blob/master/keras_applications/resnet50.py using information found in
    https://arxiv.org/pdf/1707.01629.pdf in order to adapt the resnet50
    architecture into the respective DPN50 architecture. Main differences
    are the implementation of grouped convolutions and dual path implementation,
    interpretated based on my understanding of the publication.
    Since this a modified version of ResNet50, there are no pre-trained
    weights available. Code is adapted to work with the Tensorflow-Keras
    backend and not pure Keras.
    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    global backend, layers, CARDINALITY

    if not (weights in {None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either` '
                         '`None` (random initialization),`'
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet':
        raise ValueError('Using `weights` as `"imagenet"` is not allowed.')

    # Determine proper input shape
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
        
    CARDINALITY = 32

    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = activations.relu(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [48, 48, 256], stage=2, block='a', width_increment=16, strides=(1, 1))
    x = identity_block(x, 3, [48, 48, 256], stage=2, block='b', width_increment=16)
    x = identity_block(x, 3, [48, 48, 256], stage=2, block='c', width_increment=16)

    x = conv_block(x, 3, [96, 96, 512], stage=3, block='a', width_increment=32)
    x = identity_block(x, 3, [96, 96, 512], stage=3, block='b', width_increment=32)
    x = identity_block(x, 3, [96, 96, 512], stage=3, block='c', width_increment=32)
    x = identity_block(x, 3, [96, 96, 512], stage=3, block='d', width_increment=32)

    x = conv_block(x, 3, [192, 192, 1024], stage=4, block='a', width_increment=24)
    x = identity_block(x, 3, [192, 192, 1024], stage=4, block='b', width_increment=24)
    x = identity_block(x, 3, [192, 192, 1024], stage=4, block='c', width_increment=24)
    x = identity_block(x, 3, [192, 192, 1024], stage=4, block='d', width_increment=24)
    x = identity_block(x, 3, [192, 192, 1024], stage=4, block='e', width_increment=24)
    x = identity_block(x, 3, [192, 192, 1024], stage=4, block='f', width_increment=24)

    x = conv_block(x, 3, [384, 384, 2048], stage=5, block='a', width_increment=128)
    x = identity_block(x, 3, [384, 384, 2048], stage=5, block='b', width_increment=128)
    x = identity_block(x, 3, [384, 384, 2048], stage=5, block='c', width_increment=128)
    
    x = layers.BatchNormalization(axis=bn_axis, name='bn')(x)
    x = activations.relu(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)
        else:
            warnings.warn('The output shape of `ResNet50(include_top=False)` '
                          'has been changed since Keras 2.2.0.')

    # Create model.
    model = Model(img_input, x, name='resnext50')

    # Load weights.
    if weights == 'imagenet':
        raise ValueError('Imagenet pre-trained weights are not` '
                         'available')
    elif weights is not None:
        model.load_weights(weights)

    return model
