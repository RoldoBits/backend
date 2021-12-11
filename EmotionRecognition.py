import tensorflow as tf
import numpy as np
from numba import jit, cuda
from keras.models import Model as M
from keras.preprocessing.image import ImageDataGenerator , img_to_array, load_img
from keras.applications.mobilenet import MobileNet
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, EarlyStopping

class Model:
    def s__init__(self):
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(48, 48, 3)))
        self.model.add(tf.keras.layers.Activation('relu'))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(tf.keras.layers.Dropout(0.25))
        # self.model.add(tf.keras.layers.Conv2D(64, (3, 3)))
        # self.model.add(tf.keras.layers.Activation('relu'))
        # self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        # self.model.add(tf.keras.layers.Dropout(0.25))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(512))
        # self.model.add(tf.keras.layers.Activation('relu'))
        # self.model.add(tf.keras.layers.Dropout(0.5))
        self.model.add(tf.keras.layers.Dense(7))
        self.model.add(tf.keras.layers.Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # self.model.load_weights('EmotionRecognition.h5')
        # self.model._make_predict_function()
        # self.valid
    def __init__(self):
        base_model = MobileNet( input_shape=(224,224,3), include_top= False )
        for layer in base_model.layers:
            layer.trainable = False


        x =tf.keras.layers.Flatten()(base_model.output)
        x = tf.keras.layers.Dense(units=7 , activation='softmax' )(x)

        # creating our model.
        self.model = M(base_model.input, x)
        self.model.compile(optimizer='adam', loss= categorical_crossentropy , metrics=['accuracy']  )
    # @jit(target='cuda')
    def train(self):
        es = EarlyStopping(monitor='val_accuracy', min_delta= 0.01 , patience= 5, verbose= 1, mode='auto')
        # model check point
        mc = ModelCheckpoint(filepath="best_model.h5", monitor= 'val_accuracy', verbose= 1, save_best_only= True, mode = 'auto')
        # puting call back in a list 
        call_back = [es, mc]

        # self.model.fit_generator(train_data, validation_data=validation_data, epochs=800)
        self.model.fit_generator(train_data, validation_data=validation_data,steps_per_epoch= 10,epochs= 30, validation_steps= 8,callbacks=[es,mc])
        self.model.save_weights('EmotionRecognition.h5')
    def predict(self, img):
        img = img.reshape(1, 48, 48, 1)
        return self.model.predict(img)
    def predic_from_file(self, filename):
        img = load_img(filename, target_size=(224, 224))
        img = img_to_array(img)
        # img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img, axis=0)
        img_pixels /= 255
        t=self.model.predict(img_pixels)
        max_index = np.argmax(t)
        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]
        return predicted_emotion
    def evaluate(self):
        return self.model.evaluate_generator(validation_data, 800)
    def predict_class(self, img):
        return self.model.predict_classes(img)
    def predict_prob(self, img):
        return self.model.predict_proba(img)
    def get_model(self):
        return self.model
    def get_model_json(self):
        return self.model.to_json()
    def get_model_weights(self):
        return self.model.get_weights()
    def set_model_weights(self, weights):
        self.model.set_weights(weights)
    def save_model(self, filename):        
        self.model.save_weights(filename)
    def load_model(self, filename):
        self.model.load_weights(filename)
    def get_validation_generator(self):
        return validation_generator

    def get_data_from_csv(self, csv_file):
        import csv
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            data = list(reader)
        return data
    # def download_data():
    #     !wget https://www.dropbox.com/s/nilt43hyl1dx82k/dataset.zip?dl=0

train_generator = ImageDataGenerator(
     zoom_range = 0.2, 
     shear_range = 0.2, 
     horizontal_flip=True, 
     rescale = 1./255
)

train_data = train_generator.flow_from_directory(directory= "dataset/train", 
                                               target_size=(224,224), 
                                               batch_size=32,
                                  )

validation_generator = ImageDataGenerator(rescale = 1./255 )

validation_data = validation_generator.flow_from_directory(directory= "dataset/test", 
                                           target_size=(224,224), 
                                           batch_size=32,)

c=Model()
c.train()
c.save_model('EmotionRecognition.h5')
# c.load_model('EmotionRecognition.h5')
# c.evaluate()
print(c.predic_from_file('laura2.jpg'))