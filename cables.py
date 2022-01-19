from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import sys


# Load the model
model = load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
# Replace this with the path to your image
url_image = sys.argv[1]
image = Image.open(url_image)
#resize the image to a 224x224 with the same strategy as in TM2:
#resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.ANTIALIAS)

#turn the image into a numpy array
image_array = np.asarray(image)
# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
# Load the image into the array
data[0] = normalized_image_array

# run the inference
prediction = model.predict(data)

for i in prediction:
    if i[0] > 0.8:
        print("\n..........\n\nEl cable es de color azul.\nCable directo: 4\nCable cruzado: 4\n\n..........\n")
    elif i[1] > 0.8:
        print("\n..........\n\nEl cable es de color blaco-azul.\nCable directo: 5\nCable cruzado: 5\n\n..........\n")
    elif i[2] > 0.8:
        print("\n..........\n\nEl cable es de color verde.\nCable directo: 6\nCable cruzado: 2\n\n..........\n")
    elif i[3] > 0.8:
        print("\n..........\n\nEl cable es de color blaco-verde.\nCable directo: 3\nCable cruzado: 1\n\n..........\n")
    elif i[4] > 0.8:
        print("\n..........\n\nEl cable es de color naranja.\nCable directo: 2\nCable cruzado: 6\n\n..........\n")
    elif i[5] > 0.8:
        print("\n..........\n\nEl cable es de color blaco-naranja.\nCable directo: 1\nCable cruzado: 3\n\n..........\n")
    elif i[6] > 0.8:
        print("\n..........\n\nEl cable es de color marrón.\nCable directo: 8\nCable cruzado: 8\n\n..........\n")
    elif i[7] > 0.8:
        print("\n..........\n\nEl cable es de color blaco-marrón.\nCable directo: 7\nCable cruzado: 7\n\n..........\n")
    else:
        print("\n..........\n\nNo encaja con ningún modelo.\n\n..........\n")
