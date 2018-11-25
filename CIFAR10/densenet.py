from keras import Model
from keras import backend as K
from keras.layers import Conv2D, AveragePooling2D, BatchNormalization, ReLU, Concatenate, Dense, \
    GlobalAveragePooling2D, Dropout
from keras.models import Input
from keras.regularizers import l2


def dense_block(x, blocks, growth_rate, dropout_rate=0.0, weight_decay=0.5 * 1e-4, bottleneck=False):
    for i in range(blocks):
        y = conv_block(x, growth_rate, dropout_rate, weight_decay, bottleneck)
        x = Concatenate(axis=-1)([x, y])
    return x


def bottleneck_layer(x, growth_rate, weight_decay=0.5 * 1e-4):
    x = BatchNormalization(beta_regularizer=l2(weight_decay), gamma_regularizer=l2(weight_decay))(x)
    x = ReLU()(x)
    x = Conv2D(4 * growth_rate, (1, 1), kernel_regularizer=l2(weight_decay), kernel_initializer="he_uniform",
               padding="same", use_bias=False)(x)
    return x


def composite_function(x, growth_rate, dropout_rate=0.0, weight_decay=0.5 * 1e-4):
    x = BatchNormalization(beta_regularizer=l2(weight_decay), gamma_regularizer=l2(weight_decay))(x)
    x = ReLU()(x)
    x = Conv2D(growth_rate, (3, 3), kernel_initializer="he_uniform", padding="same", use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate: x = Dropout(dropout_rate)(x)
    return x


def conv_block(x, growth_rate, dropout_rate=0.0, weight_decay=0.5 * 1e-4, bottleneck=False):
    if bottleneck:
        y = bottleneck_layer(x, growth_rate, weight_decay)
        x = composite_function(x, growth_rate, dropout_rate, weight_decay)
        x = Concatenate(axis=-1)([x, y])
    else:
        x = composite_function(x, growth_rate, dropout_rate, weight_decay)
    return x


def transition_layer(x, weight_decay=1e-4, dropout_rate=0.0, reduction=1.):
    x = BatchNormalization(beta_regularizer=l2(weight_decay), gamma_regularizer=l2(weight_decay))(x)
    x = Conv2D(int(K.int_shape(x)[3] * reduction), (1, 1), kernel_regularizer=l2(weight_decay),
               kernel_initializer="he_uniform", padding="same", use_bias=False)(x)
    if dropout_rate: x = Dropout(dropout_rate)(x)
    x = AveragePooling2D()(x)
    return x


def DenseNet(num_classes, num_filter, growth_rate, depth, dropout_rate=0.0, reduction=1., weight_decay=1e-4,
             bottleneck=False):
    N = (depth - 4) // 3
    if bottleneck:
        N //= 2
    blocks = [N, N, N]
    weight_decay *= 0.5
    inp = Input((32, 32, 3))
    x = Conv2D(num_filter, (3, 3), kernel_initializer="he_uniform", padding="same", use_bias=False,
               kernel_regularizer=l2(weight_decay))(inp)
    x = dense_block(x, blocks[0], growth_rate, dropout_rate, weight_decay)
    x = transition_layer(x, dropout_rate=dropout_rate, reduction=reduction)
    x = dense_block(x, blocks[1], growth_rate, dropout_rate, weight_decay)
    x = transition_layer(x, dropout_rate=dropout_rate, reduction=reduction)
    x = dense_block(x, blocks[2], growth_rate, dropout_rate, weight_decay)
    x = BatchNormalization(beta_regularizer=l2(weight_decay), gamma_regularizer=l2(weight_decay))(x)
    x = ReLU()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(inp, x, name='densenet')
    return model


def create_scheduler(learning_rate):
    def lr_schedule(epoch):
        lr = learning_rate
        if epoch == 150:
            lr *= 0.1
        elif epoch == 225:
            lr *= 0.1 ** 2
        return lr

    return lr_schedule

