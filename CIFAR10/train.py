import argparse

import keras
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger
from keras.datasets import cifar10
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

from densenet import DenseNet, create_scheduler

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument('--augment', type=bool, default=False, help='Augment data or not. [default: False]')
    parser.add_argument('--batch_size', type=int, default=64, help='Sets the batch size for training. [default: 64]')
    parser.add_argument('--bottleneck', type=bool, default=False, help='Add a bottleneck layer. [default: False]')
    parser.add_argument('--depth', type=int, default=40, help='Sets the depth of the model. [default: 40]')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                        help='Sets the dropout to be applied. [default: 0.2]')
    parser.add_argument('--epochs', type=int, default=300, help='Sets the batch size for training. [default: 300]')
    parser.add_argument('--growth_rate', type=int, default=12, help='Sets the growth rate. [default: 12]')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Sets the learning rate. [default: 0.1]')
    parser.add_argument('--num_classes', type=int, default=10, help='Set the number of classes')
    parser.add_argument('--reduction', type=float, default=1.0,
                        help='Sets the reduction to be applied. [default: 1.0]')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Sets the weight decay to be applied. [default: 1e-4]')
    args = parser.parse_args()

    name = 'DenseNet'
    if args.bottleneck:
        name += 'B'
    if args.reduction < 1.:
        name += 'C'
    name += str(args.depth) + str(args.growth_rate)
    num_filter = args.growth_rate * 2
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
    y_train = keras.utils.to_categorical(y_train, args.num_classes)
    y_test = keras.utils.to_categorical(y_test, args.num_classes)
    y_val = keras.utils.to_categorical(y_val, args.num_classes)

    model = DenseNet(args.num_classes, num_filter, args.growth_rate, args.depth, dropout_rate=args.dropout_rate,
                     reduction=args.reduction, weight_decay=args.weight_decay)
    scheduler = create_scheduler(args.learning_rate)
    optimizer = SGD(lr=scheduler(0), momentum=0.9, nesterov=True)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['acc'])
    model.summary()

    csv_logger = CSVLogger('log.csv', append=True, separator=';')
    checkpoint = ModelCheckpoint(filepath=name + ".h5", monitor='val_acc', verbose=1, save_best_only=True)
    lr_scheduler = LearningRateScheduler(scheduler)
    callbacks = [checkpoint, csv_logger, lr_scheduler]
    if args.augment:
        datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
    else:
        datagen = ImageDataGenerator()

    datagen.fit(x_train)

    model.fit_generator(datagen.flow(x_train, y_train, batch_size=args.batch_size), callbacks=callbacks,
                        epochs=args.epochs,
                        workers=4, steps_per_epoch=1000, validation_data=(x_val, y_val))

    score = model.evaluate(x_test, y_test, batch_size=64)
    with open('eval.txt', 'w+') as out:
        out.write(str(model.metrics) + '\n')
        out.write(str(score))
