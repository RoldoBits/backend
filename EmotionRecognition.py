import tensorflow as tf
class Model:
    def __init__(self):
        self.model = tf.Sequential()
        self.model.add(tf.Conv2D(32, (3, 3), input_shape=(48, 48, 1)))
        self.model.add(tf.Activation('relu'))
        self.model.add(tf.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(tf.Dropout(0.25))
        self.model.add(tf.Conv2D(64, (3, 3)))
        self.model.add(tf.Activation('relu'))
        self.model.add(tf.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(tf.Dropout(0.25))
        self.model.add(tf.Flatten())
        self.model.add(tf.Dense(512))
        self.model.add(tf.Activation('relu'))
        self.model.add(tf.Dropout(0.5))
        self.model.add(tf.Dense(7))
        self.model.add(tf.Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.load_weights('EmotionRecognition.h5')
        self.model._make_predict_function()
        self.valid
    def train(self):
        self.model.fit_generator(train_generator, samples_per_epoch=2000, nb_epoch=25, validation_data=validation_generator, nb_val_samples=800)
        self.model.save_weights('EmotionRecognition.h5')
    def predict(self, img):
        img = img.reshape(1, 48, 48, 1)
        return self.model.predict(img)
    def evaluate(self):
        return self.model.evaluate_generator(validation_generator, 800)
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
    def parse_data(self, data):
        #