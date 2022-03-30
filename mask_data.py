import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory


# Partie 1 : Base de données, Analyse et Préparation

DEFAULT_INPUT_SIZE = (224, 224)
BATCH_SIZE = 32
WORKING_PATH = ""
DATA_PATH = "Mask_Data/"

data = image_dataset_from_directory(WORKING_PATH + DATA_PATH, )

data_train = image_dataset_from_directory(
    WORKING_PATH + DATA_PATH, image_size = DEFAULT_INPUT_SIZE, 
    validation_split = .2, 
    subset = "training", 
    seed = 123, 
    batch_size = BATCH_SIZE)

data_test = image_dataset_from_directory(
    WORKING_PATH + DATA_PATH, image_size = DEFAULT_INPUT_SIZE, 
    validation_split = .2, 
    subset = "validation", 
    seed = 123, 
    batch_size = BATCH_SIZE)

class_names = data_train.class_names

plt.figure(figsize=(10, 10))
for images, labels in data_train.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

plt.show()


# Partie 2 : Architecture CNN sur Tensorflow

