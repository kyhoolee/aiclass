import numpy as np
import os
from keras.applications.densenet import DenseNet121, preprocess_input, decode_predictions
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from sklearn.metrics import classification_report


def evaluate(model, data_dir):
    images = []
    labels = []
    for idx, label in classes.items():
        image_files = [os.path.join(data_dir, label, name) for name in os.listdir(os.path.join(data_dir, label))]
        images.extend([img_to_array(load_img(x, target_size=(224, 224))) for x in image_files])
        labels.extend([idx] * len(image_files))
    images = np.stack(images, axis=0)
    labels = np.array(labels)

    preds = model.predict(images)
    result = np.argmax(preds, axis=1)
    print(classification_report(labels, result, target_names=[classes[i] for i in sorted(classes.keys())]))


if __name__ == "__main__":
    train_dir = "data/train"
    classes = {}
    images = []
    labels = []
    for idx, label in enumerate(os.listdir(train_dir)):
        classes[idx] = label
        image_files = [os.path.join(train_dir, label, name) for name in os.listdir(os.path.join(train_dir, label))]
        images.extend([img_to_array(load_img(x, target_size=(224, 224))) for x in image_files])
        labels.extend([idx] * len(image_files))
    images = np.stack(images, axis=0)
    labels = np.array(labels)

    base_model = DenseNet121(include_top=False)
    output = GlobalAveragePooling2D()(base_model.output)
    output = Dense(len(classes), activation="softmax")(output)
    model = Model(inputs=base_model.input, outputs=output)
    # for layer in base_model.layers:
    #     layer.trainable = False
    op = Adam(lr=0.0001)
    model.compile(optimizer=op, loss="sparse_categorical_crossentropy")

    model.fit(images, labels, batch_size=16, epochs=5, shuffle=True, verbose=1)
    evaluate(model, "data/train")
    evaluate(model, "data/val")




