import numpy as np
import os
from collections import Counter
from keras.applications.densenet import DenseNet121, preprocess_input, decode_predictions
from keras.preprocessing.image import load_img, img_to_array


if __name__ == "__main__":
    img_dir = "data/val/bees"
    images = [os.path.join(img_dir, name) for name in os.listdir(img_dir)]

    model = DenseNet121()
    img = [img_to_array(load_img(name, target_size=(224, 224))) for name in images]
    batch = np.stack(img, axis=0)
    batch = preprocess_input(batch)
    preds = model.predict(batch)
    result = decode_predictions(preds, top=1)
    print(Counter([x[0][1] for x in result]))

