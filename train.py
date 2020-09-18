import sys
from matplotlib import pyplot
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
 
def define_model():
    model = ResNet50V2(include_top=False, input_shape=(240, 320, 3))

    x = model.output
    avgpool = GlobalAveragePooling2D()(x)
    output = Dense(1, activation='sigmoid')(avgpool)

    model = Model(inputs=model.inputs, outputs=output)
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model
    
def summarize_diagnostics(history):
    # plot loss
    pyplot.subplot(211)
    print("pyplot start")
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig('/PATH_TO_SAVE_PLOT_FILE')
    pyplot.close()
    print("pyplot end")
 
# run the test harness for evaluating a model
def run_test_harness():
    
    model = define_model() # define model
    datagen = ImageDataGenerator(featurewise_center=True) # create data generator
    datagen.mean = [123.68, 116.779, 103.939] # specify imagenet mean values for centering
    
    # prepare iterator
    print("start allocate")
    train_it = datagen.flow_from_directory('/PATH_TO_TRAIN_DIR/',
        class_mode='binary', batch_size=64, target_size=(240, 320))
    test_it = datagen.flow_from_directory('/PATH_TO_TEST_DIR/',
        class_mode='binary', batch_size=64, target_size=(240, 320))
    
    # fit model
    print("start train")
    history = model.fit(train_it, steps_per_epoch=len(train_it),
        validation_data=test_it, validation_steps=len(test_it), epochs=20, verbose=1)
    model.save('/PATH_TO_MODEL_.h5_FILE')

    # evaluate model
    _, acc = model.evaluate(test_it, steps=len(test_it), verbose=0)
    print('> %.3f' % (acc * 100.0))
    # learning curves
    summarize_diagnostics(history)
    
 
# entry point, run the test harness
run_test_harness()
