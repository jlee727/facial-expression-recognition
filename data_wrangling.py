
import pandas as pd
import numpy as np
import tensorflow as tf


# read in data
data = pd.read_csv("/content/drive/My Drive/Colab Notebooks/fer2013.csv")

# split data
train = data[data.Usage == 'Training']
val = data[data.Usage == 'PublicTest']
test = data[data.Usage == 'PrivateTest']


# Change Y to one hot encoding  
def convert_emotion_to_one_hot(data):

    y = data["emotion"].values.ravel()    # flatten y
    num_classes = np.unique(y).shape[0]
    num_y = y.shape[0]
    index_offset = np.arange(num_y) * num_classes
    labels_one_hot = np.zeros((num_y, num_classes))
    labels_one_hot.flat[[index_offset + y.ravel()]] = 1

    return labels_one_hot

# Labels for classes
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# create output data
y_train = convert_emotion_to_one_hot(train)
y_val = convert_emotion_to_one_hot(val)
y_test = convert_emotion_to_one_hot(test)

# number of classes in output
y = data["emotion"].values.ravel()    # flatten y
num_classes = np.unique(y).shape[0]

# change X to workable data
def convert_pixels_to_images(data, image_size):

  pixels_values = data.pixels.str.split(" ").tolist()
  pixels_values = pd.DataFrame(pixels_values, dtype=int)

  images = pixels_values.values
  images = images.astype(np.float)
  images = images.reshape(images.shape[0], image_size, image_size, 1)
  images = images.astype('float32')

  return images

# create input data
image_size = 48
X_train = convert_pixels_to_images(train, image_size=image_size)     #images are 48*48
X_val = convert_pixels_to_images(val, image_size=image_size)         #images are 48*48
X_test = convert_pixels_to_images(test, image_size=image_size)       #images are 48*48



"""DATA AUGMENTATION (TRAINING DATA)"""

# select 200 random training images for augmentation
num_rand = 200
np.random.seed(42)
rand = list(np.random.randint(0, len(X_train), num_rand))
images_aug = X_train[rand]

# create augmentation function
def augment(image):
  image = tf.image.convert_image_dtype(image, tf.float32) # Cast and normalize the image to [0,1]
  image = tf.image.resize_with_crop_or_pad(image, image_size + 6, image_size + 6) # Add 6 pixels of padding
  image = tf.image.random_crop(image,size=[image_size, image_size, 1])             
  image = tf.image.random_flip_left_right(image)
  return image

# augment randomly selected training images images
for image in range(len(images_aug)):
  images_aug[image] = augment(images_aug[image])

# combine augmented data with training data
X_train = np.vstack((X_train, images_aug))
y_train = np.vstack((y_train, y_train[rand]))


"""DATA NORMALIZATION"""

X_train /= 255
X_val /= 255
X_test /= 255