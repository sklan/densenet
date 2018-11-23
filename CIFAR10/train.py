import keras
from keras import Model
from keras import backend as K
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger
from keras.datasets import cifar10
from keras.layers import Conv2D, AveragePooling2D, BatchNormalization, ReLU, Concatenate, Dense, \
    GlobalAveragePooling2D, Dropout
from keras.models import Input
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from sklearn.model_selection import train_test_split


def dense_block(x, blocks, growth_rate, dropout_rate=0.0, weight_decay=0.5 * 1e-4):
    for i in range(blocks):
        y = conv_block(x, growth_rate, dropout_rate, weight_decay)
        x = Concatenate(axis=-1)([x, y])
    return x


def composite_function(x, growth_rate, dropout_rate=0.0, weight_decay=0.5 * 1e-4):
    x = BatchNormalization(beta_regularizer=l2(weight_decay), gamma_regularizer=l2(weight_decay))(x)
    x = ReLU()(x)
    x = Conv2D(growth_rate, (3, 3), kernel_initializer="he_uniform", padding="same", use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate: x = Dropout(dropout_rate)(x)
    return x


def conv_block(x, growth_rate, dropout_rate=0.0, weight_decay=0.5 * 1e-4, bottleneck=False):
    x = composite_function(x, growth_rate, dropout_rate, weight_decay)
    return x


def transition_layer(x, weight_decay=1e-4, dropout_rate=0.0, reduction=1.):
    x = BatchNormalization(beta_regularizer=l2(weight_decay), gamma_regularizer=l2(weight_decay))(x)
    x = Conv2D(int(K.int_shape(x)[3] * reduction), (1, 1), kernel_regularizer=l2(weight_decay),
               kernel_initializer="he_uniform", padding="same", use_bias=False)(x)
    if dropout_rate: x = Dropout(dropout_rate)(x)
    x = AveragePooling2D()(x)
    return x


def DenseNet(classes, num_filter, growth_rate, depth, dropout_rate=0.0, reduction=1., weight_decay=0.5 * 1e-4):
    N = (depth - 4) // 3
    blocks = [N, N, N]
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
    x = Dense(classes, activation='softmax')(x)
    model = Model(inp, x, name='densenet')
    return model


def lr_schedule(epoch):
    lr = 0.1
    if epoch == 150:
        lr *= 0.1
    elif epoch == 225:
        lr *= 0.1 ** 2
    return lr


batch_size = 64
epochs = 300
num_classes = 10
growth_rate = 12
depth = 40
weight_decay = 0.5 * 1e-4
dropout_rate = 0.2
num_filter = 24

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)

model = DenseNet(num_classes, num_filter, growth_rate, depth, dropout_rate=dropout_rate)
optimizer = SGD(lr=lr_schedule(0), momentum=0.9, nesterov=True)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['acc'])
model.summary()

csv_logger = CSVLogger('log.csv', append=True, separator=';')
checkpoint = ModelCheckpoint(filepath="CIFAR10.h5", monitor='val_acc', verbose=1, save_best_only=True)
lr_scheduler = LearningRateScheduler(lr_schedule)
callbacks = [checkpoint, csv_logger, lr_scheduler]

datagen = ImageDataGenerator()
datagen.fit(x_train)
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), callbacks=callbacks, epochs=epochs,
                    workers=4, steps_per_epoch=1000, validation_data=(x_val, y_val))
score = model.evaluate(x_test, y_test, batch_size=64)
with open('eval.txt', 'w+') as out:
    out.write(str(model.metrics) + '\n')
    out.write(str(score))

