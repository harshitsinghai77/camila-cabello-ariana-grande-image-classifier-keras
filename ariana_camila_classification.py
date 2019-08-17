from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
import numpy as np
import os

model = load_model('camila_ariana.h5')
model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['acc'])
model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')
test_datagen = ImageDataGenerator(rescale=1./255)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def model_predict(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = x.reshape((1,) + x.shape)
    
    for batch in test_datagen.flow(x, batch_size=1):
        pred = model.predict(batch)
        if pred > 0.5:
            return 'Camila Cabello'
        else:
            return 'Ariana Grande'
        break
    return pred

