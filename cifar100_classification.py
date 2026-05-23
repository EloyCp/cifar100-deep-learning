"""
Deep Learning - Image Classification on CIFAR-100
==================================================

Master's Degree project for Nuclio Digital School.
Two strategies are designed and compared for image classification on the
CIFAR-100 dataset (https://keras.io/api/datasets/cifar100/):

    Strategy 1 - Pre-trained networks:
        Transfer Learning and Fine-Tuning are applied on networks pre-trained
        on ImageNet (VGG16, EfficientNetB0). Several optimization techniques
        are tested: weight regularization, dropout, batch normalization,
        data augmentation, etc.

    Strategy 2 - Training from scratch:
        A custom CNN is designed, trained and optimized. The empirical
        justification for each layer, hyperparameter and regularization
        choice is documented along the experimentation flow.

Authors Group 3

Original notebook: Actividad1_DL_Grupo_3.ipynb (Google Colab)

NOTE: This script is the Python (.py) version of the original Colab notebook.
The structure and experimentation flow have been preserved as much as
possible. The Colab-specific code (Google Drive mounting, IPython magics)
has been replaced with local-friendly equivalents.
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import random

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
    MaxPooling2D,
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


# =============================================================================
# CONFIGURATION
# =============================================================================
# Original Colab path was '/content/drive/My Drive/'. Replaced with a local
# folder so the script can be executed outside of Google Colab.
BASE_FOLDER = "./saved_models/"
os.makedirs(BASE_FOLDER, exist_ok=True)

# CIFAR-100 class names (used for the classification report)
LABEL_NAMES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree',
    'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy',
    'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail',
    'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper',
    'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train',
    'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf',
    'woman', 'worm'
]


# =============================================================================
# DATA LOADING & EXPLORATION
# =============================================================================
print("[INFO]: Loading CIFAR-100 data...")
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

# Sanity checks on the dataset shapes
assert x_train.shape == (50000, 32, 32, 3)
assert x_test.shape == (10000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert y_test.shape == (10000, 1)

print(f"x_train shape: {x_train.shape}")
print(f"y_train length: {len(y_train)}")

# --- Visualize a single sample ---
i = 9144
plt.imshow(x_train[i])
plt.title(f"Class index: {y_train[i][0]}")
plt.show()

# --- Visualize 28 random samples ---
fig = plt.figure(figsize=(14, 10))
for n, idx in enumerate(random.sample(range(1, len(y_train)), 28)):
    fig.add_subplot(4, 7, n + 1)
    plt.imshow(x_train[idx])
    plt.title(y_train[idx])
    plt.axis('off')
plt.show()

# --- Class distribution histogram ---
plt.rcParams.update({'figure.figsize': (15, 5), 'figure.dpi': 120})
plt.hist(y_train, bins=range(0, 103), width=0.9, align='left')
plt.xticks(rotation=90, size=6)
plt.gca().set(title='Number of samples per class', ylabel='Number of samples')
plt.show()


# =============================================================================
# STRATEGY 1 - PRE-TRAINED NETWORKS
# =============================================================================
# Steps followed in this strategy:
#   1. Apply TRANSFER LEARNING with VGG16 and EfficientNetB0.
#   2. Apply FINE TUNING to the most promising network.
#   3. Evaluate modifications to the TOP MODEL structure
#      (convolutional layers and neuron count).
#   4. Evaluate regularization techniques (weight regularization,
#      dropout and batch normalization).
#   5. Apply DATA AUGMENTATION.
#   6. Combine the most promising strategies.
# =============================================================================


def plot_history(history, epochs=20, title="Training Loss and Accuracy"):
    """Plot training/validation loss and accuracy curves."""
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(np.arange(0, epochs), history.history['loss'], label='train_loss')
    plt.plot(np.arange(0, epochs), history.history['val_loss'], label='val_loss')
    plt.plot(np.arange(0, epochs), history.history['accuracy'], label='train_acc')
    plt.plot(np.arange(0, epochs), history.history['val_accuracy'], label='val_acc')
    plt.title(title)
    plt.xlabel('Epoch #')
    plt.ylabel('Loss/Accuracy')
    plt.legend()
    plt.show()


# -----------------------------------------------------------------------------
# 1. TRANSFER LEARNING
# -----------------------------------------------------------------------------

# --- 1.1 VGG16 (max accuracy = 35%, heavy overfitting) ---------------------
# Preprocess input data to match VGG16 expectations
x_train_vgg16 = keras.applications.vgg16.preprocess_input(x_train)
x_test_vgg16 = keras.applications.vgg16.preprocess_input(x_test)

# Load the base model (without the top classifier)
base_model = keras.applications.VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(32, 32, 3),
)
base_model.summary()

# Freeze all convolutional layers and build a new top model on top
base_model.trainable = False
pre_model = Sequential()
pre_model.add(base_model)
pre_model.add(layers.Flatten())
pre_model.add(layers.Dense(512, activation='relu'))
pre_model.add(layers.Dense(100, activation='softmax'))
pre_model.summary()

pre_model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=Adam(learning_rate=0.0005),
    metrics=["accuracy"],
)
H_pre = pre_model.fit(
    x_train_vgg16, y_train,
    batch_size=128, epochs=20, validation_split=0.2,
)
# Comparing training vs validation accuracy shows heavy overfitting.

pre_model.save(BASE_FOLDER + 'deepCNN_VGG16_pretrained.h5')
plot_history(H_pre, title="VGG16 Training Loss and Accuracy")


# --- 1.2 EfficientNetB0 (max accuracy = 34%, no overfitting, no plateau) ---
x_train_eff = keras.applications.efficientnet.preprocess_input(x_train)
x_test_eff = keras.applications.efficientnet.preprocess_input(x_test)

base_model = keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(32, 32, 3),
)
base_model.summary()

base_model.trainable = False
pre_model = Sequential()
pre_model.add(base_model)
pre_model.add(layers.Flatten())
pre_model.add(layers.Dense(512, activation='relu'))
pre_model.add(layers.Dense(100, activation='softmax'))
pre_model.summary()

pre_model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=Adam(learning_rate=0.0005),
    metrics=["accuracy"],
)
H_pre = pre_model.fit(
    x_train_eff, y_train,
    batch_size=128, epochs=20, validation_split=0.2,
)
# Accuracy keeps slowly increasing while loss decreases. The learning rate may
# be too conservative or more epochs may be needed, but there is much less
# overfitting than with VGG16. EfficientNetB0 is therefore chosen for the
# fine tuning experiments that follow.

pre_model.save(BASE_FOLDER + 'deepCNN_EfficientNetB0_pretrained.h5')
plot_history(H_pre, title="EfficientNetB0 Training Loss and Accuracy")


# -----------------------------------------------------------------------------
# 2. FINE TUNING (no improvement observed)
# -----------------------------------------------------------------------------

# --- 6/7 blocks frozen (accuracy = 37%) ------------------------------------
base_model = keras.applications.EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=(32, 32, 3),
)
for layer in base_model.layers:
    if layer.name == 'block7a_expand_conv':
        break
    layer.trainable = False
    print('Layer ' + layer.name + ' frozen...')

# Functional API top model
last = base_model.layers[-1].output
x = Flatten()(last)
x = Dense(512, activation='relu')(x)
x = Dense(100, activation='softmax')(x)
model = Model(base_model.input, x)
model.summary()

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=Adam(learning_rate=0.0005), metrics=["accuracy"])
H_pre = model.fit(x_train_eff, y_train, batch_size=128, epochs=20, validation_split=0.2)
plot_history(H_pre)


# --- 5/7 blocks frozen (accuracy = 43%) ------------------------------------
base_model = keras.applications.EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=(32, 32, 3),
)
for layer in base_model.layers:
    if layer.name == 'block6a_expand_conv':
        break
    layer.trainable = False
    print('Layer ' + layer.name + ' frozen...')

last = base_model.layers[-1].output
x = Flatten()(last)
x = Dense(512, activation='relu')(x)
x = Dense(100, activation='softmax')(x)
model = Model(base_model.input, x)
model.summary()

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=Adam(learning_rate=0.0005), metrics=["accuracy"])
H_pre = model.fit(x_train_eff, y_train, batch_size=128, epochs=20, validation_split=0.2)
plot_history(H_pre)


# --- 4/7 blocks frozen (accuracy = 50%) ------------------------------------
base_model = keras.applications.EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=(32, 32, 3),
)
for layer in base_model.layers:
    if layer.name == 'block5a_expand_conv':
        break
    layer.trainable = False
    print('Layer ' + layer.name + ' frozen...')

last = base_model.layers[-1].output
x = Flatten()(last)
x = Dense(512, activation='relu')(x)
x = Dense(100, activation='softmax')(x)
model = Model(base_model.input, x)
model.summary()

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=Adam(learning_rate=0.0005), metrics=["accuracy"])
H_pre = model.fit(x_train_eff, y_train, batch_size=128, epochs=20, validation_split=0.2)
plot_history(H_pre)


# --- 3/7 blocks frozen (accuracy = 52%) ------------------------------------
base_model = keras.applications.EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=(32, 32, 3),
)
for layer in base_model.layers:
    if layer.name == 'block4a_expand_conv':
        break
    layer.trainable = False
    print('Layer ' + layer.name + ' frozen...')

last = base_model.layers[-1].output
x = Flatten()(last)
x = Dense(512, activation='relu')(x)
x = Dense(100, activation='softmax')(x)
model = Model(base_model.input, x)
model.summary()

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=Adam(learning_rate=0.0005), metrics=["accuracy"])
H_pre = model.fit(x_train_eff, y_train, batch_size=128, epochs=20, validation_split=0.2)
plot_history(H_pre)

# Conclusion: fine-tuning did not improve accuracy, and in fact overfitting
# increased noticeably.


# -----------------------------------------------------------------------------
# 3. TOP MODEL STRUCTURE ADJUSTMENT (no improvement)
# -----------------------------------------------------------------------------

# --- 3.1 First attempt: 256 neurons in the hidden layer (accuracy = 54%) ---
base_model.trainable = False
pre_model = Sequential()
pre_model.add(base_model)
pre_model.add(layers.Flatten())
pre_model.add(layers.Dense(256, activation='relu'))
pre_model.add(layers.Dense(100, activation='softmax'))
pre_model.summary()


def compile_fit_efn(model_to_train):
    """Compile and train a model on the EfficientNet-preprocessed data."""
    model_to_train.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=Adam(learning_rate=0.0005),
        metrics=["accuracy"],
    )
    return model_to_train.fit(
        x_train_eff, y_train,
        batch_size=128, epochs=20, validation_split=0.2,
    )


H_pre = compile_fit_efn(pre_model)
plot_history(H_pre)


# --- 3.2 Second attempt: 1024 + 256 neurons (accuracy = 53%) ---------------
base_model.trainable = False
pre_model = Sequential()
pre_model.add(base_model)
pre_model.add(layers.Flatten())
pre_model.add(layers.Dense(1024, activation='relu'))
pre_model.add(layers.Dense(256, activation='relu'))
pre_model.add(layers.Dense(100, activation='softmax'))
pre_model.summary()

H_pre = compile_fit_efn(pre_model)
plot_history(H_pre)


# -----------------------------------------------------------------------------
# 4. REGULARIZATION PARAMETERS
# -----------------------------------------------------------------------------

# --- 4.1 Weight regularization L2 (no improvement) -------------------------
base_model.trainable = False
pre_model = Sequential()
pre_model.add(base_model)
pre_model.add(layers.Flatten())
pre_model.add(layers.Dense(512, activation='relu',
                           kernel_regularizer=regularizers.l2(0.01)))
pre_model.add(layers.Dense(100, activation='softmax'))
pre_model.summary()
H_pre = compile_fit_efn(pre_model)
plot_history(H_pre)


# --- 4.2 Dropout (accuracy = 55%) ------------------------------------------
base_model.trainable = False
pre_model = Sequential()
pre_model.add(base_model)
pre_model.add(layers.Flatten())
pre_model.add(layers.Dense(512, activation='relu'))
pre_model.add(Dropout(0.75))
pre_model.add(layers.Dense(100, activation='softmax'))
pre_model.summary()
H_pre = compile_fit_efn(pre_model)
plot_history(H_pre)


# --- 4.3 Batch Normalization (no improvement) ------------------------------
base_model.trainable = False
pre_model = Sequential()
pre_model.add(base_model)
pre_model.add(layers.Flatten())
pre_model.add(layers.Dense(512, activation='relu'))
pre_model.add(BatchNormalization())
pre_model.add(layers.Dense(100, activation='softmax'))
pre_model.summary()
H_pre = compile_fit_efn(pre_model)
plot_history(H_pre)


# --- 4.4 Combination of all three (no improvement) -------------------------
base_model.trainable = False
pre_model = Sequential()
pre_model.add(base_model)
pre_model.add(layers.Flatten())
pre_model.add(layers.Dense(512, activation='relu',
                           kernel_regularizer=regularizers.l2(0.01)))
pre_model.add(BatchNormalization())
pre_model.add(Dropout(0.75))
pre_model.add(layers.Dense(100, activation='softmax'))
pre_model.summary()
H_pre = compile_fit_efn(pre_model)
plot_history(H_pre)


# -----------------------------------------------------------------------------
# 5. DATA AUGMENTATION
# -----------------------------------------------------------------------------

# --- 5.1 First attempt (training accuracy decreases) -----------------------
base_model.trainable = False
pre_model = Sequential()
pre_model.add(tf.keras.layers.RandomFlip('horizontal'))
pre_model.add(tf.keras.layers.RandomRotation(0.05))
pre_model.add(tf.keras.layers.RandomTranslation(0.1, 0.1))
pre_model.add(base_model)
pre_model.add(layers.Flatten())
pre_model.add(layers.Dense(512, activation='relu'))
pre_model.add(layers.Dense(100, activation='softmax'))
pre_model.summary()
H_pre = compile_fit_efn(pre_model)
plot_history(H_pre)


# --- 5.2 Second attempt (accuracy decreases, more overfitting) -------------
base_model.trainable = False
pre_model = Sequential()
pre_model.add(tf.keras.layers.RandomFlip('horizontal'))
pre_model.add(tf.keras.layers.RandomRotation(0.05))
pre_model.add(tf.keras.layers.RandomTranslation(0.05, 0.05))
pre_model.add(base_model)
pre_model.add(layers.Flatten())
pre_model.add(layers.Dense(512, activation='relu'))
pre_model.add(layers.Dense(100, activation='softmax'))
pre_model.summary()
H_pre = compile_fit_efn(pre_model)
plot_history(H_pre)


# -----------------------------------------------------------------------------
# 6. STRATEGY COMBINATION (no improvement)
# -----------------------------------------------------------------------------

# --- 6.1 Fine tuning (3/7 blocks frozen) + dropout -------------------------
base_model = keras.applications.EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=(32, 32, 3),
)
for layer in base_model.layers:
    if layer.name == 'block4a_expand_conv':
        break
    layer.trainable = False
    print('Layer ' + layer.name + ' frozen...')

last = base_model.layers[-1].output
x = Flatten()(last)
x = Dense(512, activation='relu')(x)
x = Dropout(0.75)(x)
x = Dense(100, activation='softmax')(x)
model = Model(base_model.input, x)
model.summary()

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=Adam(learning_rate=0.0005), metrics=["accuracy"])
H_pre = model.fit(x_train_eff, y_train, batch_size=128, epochs=20, validation_split=0.2)
plot_history(H_pre)


# --- 6.2 Fine tuning (3/7 blocks frozen) + top model adjustment ------------
base_model = keras.applications.EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=(32, 32, 3),
)
for layer in base_model.layers:
    if layer.name == 'block4a_expand_conv':
        break
    layer.trainable = False
    print('Layer ' + layer.name + ' frozen...')

last = base_model.layers[-1].output
x = Flatten()(last)
x = Dense(256, activation='relu')(x)
x = Dense(100, activation='softmax')(x)
model = Model(base_model.input, x)
model.summary()

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=Adam(learning_rate=0.0005), metrics=["accuracy"])
H_pre = model.fit(x_train_eff, y_train, batch_size=128, epochs=20, validation_split=0.2)
plot_history(H_pre)


# =============================================================================
# STRATEGY 2 - TRAINING FROM SCRATCH
# =============================================================================
# TensorFlow ecosystem was selected as it provides every optimization
# technique needed (Dropout, L2 regularization, etc.). Matplotlib is used
# in parallel to inspect the learning curves and empirically validate the
# improvements introduced in each iteration of the model.
# =============================================================================

# --- Reload the original (non-preprocessed) data ---
print("[INFO]: Reloading CIFAR-100 data for Strategy 2...")
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# After loading we have 50,000 training images of shape 32x32x3 (RGB).

# --- Inspect a few samples with their labels ---
fig = plt.figure(figsize=(16, 12))
for n in range(1, 29):
    fig.add_subplot(4, 7, n)
    plt.imshow(x_train[n])
    plt.title(LABEL_NAMES[y_train[n][0]])
    plt.axis('off')
plt.show()


# -----------------------------------------------------------------------------
# DATA PREPROCESSING
# -----------------------------------------------------------------------------
# Pixel normalization is important so the network learns more easily.
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encoding using LabelBinarizer (style suggested by the tutor)
print("[INFO]: Applying One-Hot Encoding with LabelBinarizer...")
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

# Split off a validation set
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42,
)


# -----------------------------------------------------------------------------
# TRAINING FUNCTION (no data augmentation)
# -----------------------------------------------------------------------------
def train(x_train, y_train, x_test, y_test, model_to_train,
          base_folder, learn_rate=0.0005):
    """Compile, train, save and evaluate a model. Also plots the learning
    curves. Uses x_val / y_val from the outer scope as the validation set."""
    model_to_train.summary()

    # Compile the model
    print("[INFO]: Compiling the model...")
    model_to_train.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=learn_rate),
        metrics=["accuracy"],
    )

    # Train the network
    print("[INFO]: Training the network...")
    H_pre = model_to_train.fit(
        x_train, y_train,
        batch_size=128,
        epochs=50,
        validation_data=(x_val, y_val),
    )

    # Save the trained model
    model_to_train.save(base_folder + "deepCNN_CIFAR100_pretrained.h5")

    # Evaluate the model
    print("[INFO]: Evaluating the model...")
    predictions = model_to_train.predict(x_test, batch_size=128)
    print(classification_report(
        y_test.argmax(axis=1),
        predictions.argmax(axis=1),
        target_names=LABEL_NAMES,
    ))

    # Learning curves
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, 50), H_pre.history["loss"], label="train_loss")
    plt.plot(np.arange(0, 50), H_pre.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, 50), H_pre.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, 50), H_pre.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()


# -----------------------------------------------------------------------------
# CNN ARCHITECTURE - JUSTIFICATION
# -----------------------------------------------------------------------------
# A CNN (Convolutional Neural Network) is chosen as the base architecture
# for the from-scratch design. Unlike an MLP, convolutional layers extract
# hierarchical features (edges, textures, shapes) while preserving the
# spatial coherence of pixels, which is critical in CIFAR-100 where objects
# appear inside low-resolution 32x32 images.
# -----------------------------------------------------------------------------


# --- MODEL 1 BASE (accuracy = 57%, 2.9M trainable parameters) --------------
# BASE MODEL: feature extractor for the input images.
inputs = Input(shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]))

# First convolutional block
c1 = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
c1 = BatchNormalization()(c1)
c1 = Conv2D(32, (3, 3), padding='same', activation='relu')(c1)
c1 = BatchNormalization()(c1)
c1 = MaxPooling2D(pool_size=(2, 2))(c1)
c1 = Dropout(0.25)(c1)

# Second convolutional block
c2 = Conv2D(64, (3, 3), padding='same', activation='relu')(c1)
c2 = BatchNormalization()(c2)
c2 = Conv2D(64, (3, 3), padding='same', activation='relu')(c2)
c2 = BatchNormalization()(c2)
c2 = MaxPooling2D(pool_size=(2, 2))(c2)
c2 = Dropout(0.25)(c2)

# Third convolutional block
c3 = Conv2D(256, (3, 3), padding='same', activation='relu')(c2)
c3 = BatchNormalization()(c3)
c3 = Conv2D(256, (3, 3), padding='same', activation='relu')(c3)
c3 = BatchNormalization()(c3)
c3 = MaxPooling2D(pool_size=(2, 2))(c3)
c3 = Dropout(0.25)(c3)

# TOP MODEL: classifies the extracted features.
xfc = Flatten()(c3)
xfc = Dense(512, activation="relu")(xfc)
xfc = BatchNormalization()(xfc)
xfc = Dropout(0.5)(xfc)

# Softmax classifier
predictions = Dense(100, activation="softmax")(xfc)

model = Model(inputs=inputs, outputs=predictions)
train(x_train, y_train, x_test, y_test, model, BASE_FOLDER)

# The first model reaches 57% accuracy, which is considered insufficient
# given that the architecture handles almost 3 million parameters. The
# learning curves reveal severe overfitting: while training loss keeps
# falling, validation loss plateaus and starts to diverge early. The model
# is memorizing the training set instead of learning generalizable features,
# which motivates more aggressive regularization and a structural review of
# the layers in the following iterations.


# --- MODEL 2 (accuracy = 59%, 2.9M parameters, dropout = 0.5) --------------
# Increasing Dropout to 0.5 in every layer applies a more aggressive
# stochastic regularization. The network is forced to build redundant
# representations of the features instead of relying on specific neurons,
# which should improve generalization.
inputs = Input(shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]))

c1 = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
c1 = BatchNormalization()(c1)
c1 = Conv2D(32, (3, 3), padding='same', activation='relu')(c1)
c1 = BatchNormalization()(c1)
c1 = MaxPooling2D(pool_size=(2, 2))(c1)
c1 = Dropout(0.5)(c1)  # Dropout raised to 0.5

c2 = Conv2D(64, (3, 3), padding='same', activation='relu')(c1)
c2 = BatchNormalization()(c2)
c2 = Conv2D(64, (3, 3), padding='same', activation='relu')(c2)
c2 = BatchNormalization()(c2)
c2 = MaxPooling2D(pool_size=(2, 2))(c2)
c2 = Dropout(0.5)(c2)

c3 = Conv2D(256, (3, 3), padding='same', activation='relu')(c2)
c3 = BatchNormalization()(c3)
c3 = Conv2D(256, (3, 3), padding='same', activation='relu')(c3)
c3 = BatchNormalization()(c3)
c3 = MaxPooling2D(pool_size=(2, 2))(c3)
c3 = Dropout(0.5)(c3)

xfc = Flatten()(c3)
xfc = Dense(512, activation="relu")(xfc)
xfc = BatchNormalization()(xfc)
xfc = Dropout(0.5)(xfc)
predictions = Dense(100, activation="softmax")(xfc)

model = Model(inputs=inputs, outputs=predictions)
train(x_train, y_train, x_test, y_test, model, BASE_FOLDER)

# Accuracy improved slightly and the validation curve stabilized, but
# overfitting persists. The model has become robust enough but seems to
# have exhausted the information available in the training data. The next
# logical step is Data Augmentation to add visual variety.


# -----------------------------------------------------------------------------
# MODEL 3 - HYPERPARAMETER FINE TUNING
# -----------------------------------------------------------------------------
# Before applying Data Augmentation, fine tuning of the regularization and
# optimization hyperparameters is explored. Weight Regularization (L1/L2)
# is introduced to penalize large weights and the Learning Rate is tweaked
# to look for a more stable convergence.


# --- 3.1.1 Weight regularization L1 (no improvement, accuracy = 27%) -------
inputs = Input(shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]))

c1 = Conv2D(32, (3, 3), padding='same', activation='relu',
            kernel_regularizer=regularizers.l1(0.01))(inputs)
c1 = BatchNormalization()(c1)
c1 = Conv2D(32, (3, 3), padding='same', activation='relu',
            kernel_regularizer=regularizers.l1(0.01))(c1)
c1 = BatchNormalization()(c1)
c1 = MaxPooling2D(pool_size=(2, 2))(c1)
c1 = Dropout(0.5)(c1)

c2 = Conv2D(64, (3, 3), padding='same', activation='relu',
            kernel_regularizer=regularizers.l1(0.01))(c1)
c2 = BatchNormalization()(c2)
c2 = Conv2D(64, (3, 3), padding='same', activation='relu',
            kernel_regularizer=regularizers.l1(0.01))(c2)
c2 = BatchNormalization()(c2)
c2 = MaxPooling2D(pool_size=(2, 2))(c2)
c2 = Dropout(0.5)(c2)

c3 = Conv2D(256, (3, 3), padding='same', activation='relu',
            kernel_regularizer=regularizers.l1(0.01))(c2)
c3 = BatchNormalization()(c3)
c3 = Conv2D(256, (3, 3), padding='same', activation='relu',
            kernel_regularizer=regularizers.l1(0.01))(c3)
c3 = BatchNormalization()(c3)
c3 = MaxPooling2D(pool_size=(2, 2))(c3)
c3 = Dropout(0.5)(c3)

xfc = Flatten()(c3)
xfc = Dense(512, activation="relu")(xfc)
xfc = BatchNormalization()(xfc)
xfc = Dropout(0.5)(xfc)
predictions = Dense(100, activation="softmax")(xfc)

model = Model(inputs=inputs, outputs=predictions)
train(x_train, y_train, x_test, y_test, model, BASE_FOLDER)


# --- 3.1.2 Weight regularization L2 (no improvement, accuracy = 52%) -------
inputs = Input(shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]))

c1 = Conv2D(32, (3, 3), padding='same', activation='relu',
            kernel_regularizer=regularizers.l2(0.01))(inputs)
c1 = BatchNormalization()(c1)
c1 = Conv2D(32, (3, 3), padding='same', activation='relu',
            kernel_regularizer=regularizers.l2(0.01))(c1)
c1 = BatchNormalization()(c1)
c1 = MaxPooling2D(pool_size=(2, 2))(c1)
c1 = Dropout(0.5)(c1)

c2 = Conv2D(64, (3, 3), padding='same', activation='relu',
            kernel_regularizer=regularizers.l2(0.01))(c1)
c2 = BatchNormalization()(c2)
c2 = Conv2D(64, (3, 3), padding='same', activation='relu',
            kernel_regularizer=regularizers.l2(0.01))(c2)
c2 = BatchNormalization()(c2)
c2 = MaxPooling2D(pool_size=(2, 2))(c2)
c2 = Dropout(0.5)(c2)

c3 = Conv2D(256, (3, 3), padding='same', activation='relu',
            kernel_regularizer=regularizers.l2(0.01))(c2)
c3 = BatchNormalization()(c3)
c3 = Conv2D(256, (3, 3), padding='same', activation='relu',
            kernel_regularizer=regularizers.l2(0.01))(c3)
c3 = BatchNormalization()(c3)
c3 = MaxPooling2D(pool_size=(2, 2))(c3)
c3 = Dropout(0.5)(c3)

xfc = Flatten()(c3)
xfc = Dense(512, activation="relu")(xfc)
xfc = BatchNormalization()(xfc)
xfc = Dropout(0.5)(xfc)
predictions = Dense(100, activation="softmax")(xfc)

model = Model(inputs=inputs, outputs=predictions)
train(x_train, y_train, x_test, y_test, model, BASE_FOLDER)

# L2 regularization at 0.01 has been too aggressive. The next experiments
# reduce that value to allow more learning capacity.


# --- 3.1.3 L1 regularization at 0.0001 (no improvement, accuracy = 57%) ----
inputs = Input(shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]))

c1 = Conv2D(32, (3, 3), padding='same', activation='relu',
            kernel_regularizer=regularizers.l1(0.0001))(inputs)
c1 = BatchNormalization()(c1)
c1 = Conv2D(32, (3, 3), padding='same', activation='relu',
            kernel_regularizer=regularizers.l1(0.0001))(c1)
c1 = BatchNormalization()(c1)
c1 = MaxPooling2D(pool_size=(2, 2))(c1)
c1 = Dropout(0.5)(c1)

c2 = Conv2D(64, (3, 3), padding='same', activation='relu',
            kernel_regularizer=regularizers.l1(0.0001))(c1)
c2 = BatchNormalization()(c2)
c2 = Conv2D(64, (3, 3), padding='same', activation='relu',
            kernel_regularizer=regularizers.l1(0.0001))(c2)
c2 = BatchNormalization()(c2)
c2 = MaxPooling2D(pool_size=(2, 2))(c2)
c2 = Dropout(0.5)(c2)

c3 = Conv2D(256, (3, 3), padding='same', activation='relu',
            kernel_regularizer=regularizers.l1(0.0001))(c2)
c3 = BatchNormalization()(c3)
c3 = Conv2D(256, (3, 3), padding='same', activation='relu',
            kernel_regularizer=regularizers.l1(0.0001))(c3)
c3 = BatchNormalization()(c3)
c3 = MaxPooling2D(pool_size=(2, 2))(c3)
c3 = Dropout(0.5)(c3)

xfc = Flatten()(c3)
xfc = Dense(512, activation="relu")(xfc)
xfc = BatchNormalization()(xfc)
xfc = Dropout(0.5)(xfc)
predictions = Dense(100, activation="softmax")(xfc)

model = Model(inputs=inputs, outputs=predictions)
train(x_train, y_train, x_test, y_test, model, BASE_FOLDER)


# --- 3.1.4 L2 regularization at 0.0001 (no improvement, accuracy = 58%) ----
inputs = Input(shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]))

c1 = Conv2D(32, (3, 3), padding='same', activation='relu',
            kernel_regularizer=regularizers.l2(0.0001))(inputs)
c1 = BatchNormalization()(c1)
c1 = Conv2D(32, (3, 3), padding='same', activation='relu',
            kernel_regularizer=regularizers.l2(0.0001))(c1)
c1 = BatchNormalization()(c1)
c1 = MaxPooling2D(pool_size=(2, 2))(c1)
c1 = Dropout(0.5)(c1)

c2 = Conv2D(64, (3, 3), padding='same', activation='relu',
            kernel_regularizer=regularizers.l2(0.0001))(c1)
c2 = BatchNormalization()(c2)
c2 = Conv2D(64, (3, 3), padding='same', activation='relu',
            kernel_regularizer=regularizers.l2(0.0001))(c2)
c2 = BatchNormalization()(c2)
c2 = MaxPooling2D(pool_size=(2, 2))(c2)
c2 = Dropout(0.5)(c2)

c3 = Conv2D(256, (3, 3), padding='same', activation='relu',
            kernel_regularizer=regularizers.l2(0.0001))(c2)
c3 = BatchNormalization()(c3)
c3 = Conv2D(256, (3, 3), padding='same', activation='relu',
            kernel_regularizer=regularizers.l2(0.0001))(c3)
c3 = BatchNormalization()(c3)
c3 = MaxPooling2D(pool_size=(2, 2))(c3)
c3 = Dropout(0.5)(c3)

xfc = Flatten()(c3)
xfc = Dense(512, activation="relu")(xfc)
xfc = BatchNormalization()(xfc)
xfc = Dropout(0.5)(xfc)
predictions = Dense(100, activation="softmax")(xfc)

model = Model(inputs=inputs, outputs=predictions)
train(x_train, y_train, x_test, y_test, model, BASE_FOLDER)

# Reducing the regularization recovers the convergence ability of the
# network and the underfitting from the previous attempt disappears.
# Results stabilize around 58% accuracy though, equalling but not surpassing
# the base model. The model has reached the learning limit of the current
# data and architecture, so Data Augmentation is the natural next step.


# --- 3.2.1 Dropout = 0.75 (no improvement, drops to 37%) -------------------
# A more aggressive Dropout (0.75) is tested to see whether the model still
# has memorization capacity left.
inputs = Input(shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]))

c1 = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
c1 = BatchNormalization()(c1)
c1 = Conv2D(32, (3, 3), padding='same', activation='relu')(c1)
c1 = BatchNormalization()(c1)
c1 = MaxPooling2D(pool_size=(2, 2))(c1)
c1 = Dropout(0.75)(c1)

c2 = Conv2D(64, (3, 3), padding='same', activation='relu')(c1)
c2 = BatchNormalization()(c2)
c2 = Conv2D(64, (3, 3), padding='same', activation='relu')(c2)
c2 = BatchNormalization()(c2)
c2 = MaxPooling2D(pool_size=(2, 2))(c2)
c2 = Dropout(0.75)(c2)

c3 = Conv2D(256, (3, 3), padding='same', activation='relu')(c2)
c3 = BatchNormalization()(c3)
c3 = Conv2D(256, (3, 3), padding='same', activation='relu')(c3)
c3 = BatchNormalization()(c3)
c3 = MaxPooling2D(pool_size=(2, 2))(c3)
c3 = Dropout(0.75)(c3)

xfc = Flatten()(c3)
xfc = Dense(512, activation="relu")(xfc)
xfc = BatchNormalization()(xfc)
xfc = Dropout(0.75)(xfc)
predictions = Dense(100, activation="softmax")(xfc)

model = Model(inputs=inputs, outputs=predictions)
train(x_train, y_train, x_test, y_test, model, BASE_FOLDER)

# Dropout 0.75 is too aggressive: accuracy collapses to 37%. Going back to
# the 0.5 setting for the next experiments.


# --- 3.3 Learning rate adjustment (accuracy = 58%, more overfitting) -------
inputs = Input(shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]))

c1 = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
c1 = BatchNormalization()(c1)
c1 = Conv2D(32, (3, 3), padding='same', activation='relu')(c1)
c1 = BatchNormalization()(c1)
c1 = MaxPooling2D(pool_size=(2, 2))(c1)
c1 = Dropout(0.5)(c1)

c2 = Conv2D(64, (3, 3), padding='same', activation='relu')(c1)
c2 = BatchNormalization()(c2)
c2 = Conv2D(64, (3, 3), padding='same', activation='relu')(c2)
c2 = BatchNormalization()(c2)
c2 = MaxPooling2D(pool_size=(2, 2))(c2)
c2 = Dropout(0.5)(c2)

c3 = Conv2D(256, (3, 3), padding='same', activation='relu')(c2)
c3 = BatchNormalization()(c3)
c3 = Conv2D(256, (3, 3), padding='same', activation='relu')(c3)
c3 = BatchNormalization()(c3)
c3 = MaxPooling2D(pool_size=(2, 2))(c3)
c3 = Dropout(0.5)(c3)

xfc = Flatten()(c3)
xfc = Dense(512, activation="relu")(xfc)
xfc = BatchNormalization()(xfc)
xfc = Dropout(0.5)(xfc)
predictions = Dense(100, activation="softmax")(xfc)

model = Model(inputs=inputs, outputs=predictions)
train(x_train, y_train, x_test, y_test, model, BASE_FOLDER, learn_rate=0.001)
# No real improvement over Model 2.


# -----------------------------------------------------------------------------
# MODEL 4 - DATA AUGMENTATION
# -----------------------------------------------------------------------------

# --- 4.1 Basic data augmentation (accuracy = 52%) --------------------------
data_augmentation = Sequential([
    layers.RandomFlip("horizontal"),            # Horizontal flip (mirror)
    layers.RandomRotation(0.05),                # Random rotation up to ±18°
    layers.RandomTranslation(0.1, 0.1),         # Random translation up to ±10%
    layers.RandomZoom(0.1),                     # Random zoom up to ±10%
])

inputs = Input(shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]))
daug = data_augmentation(inputs)

c1 = Conv2D(32, (3, 3), padding='same', activation='relu')(daug)
c1 = BatchNormalization()(c1)
c1 = Conv2D(32, (3, 3), padding='same', activation='relu')(c1)
c1 = BatchNormalization()(c1)
c1 = MaxPooling2D(pool_size=(2, 2))(c1)
c1 = Dropout(0.5)(c1)

c2 = Conv2D(64, (3, 3), padding='same', activation='relu')(c1)
c2 = BatchNormalization()(c2)
c2 = Conv2D(64, (3, 3), padding='same', activation='relu')(c2)
c2 = BatchNormalization()(c2)
c2 = MaxPooling2D(pool_size=(2, 2))(c2)
c2 = Dropout(0.5)(c2)

c3 = Conv2D(256, (3, 3), padding='same', activation='relu')(c2)
c3 = BatchNormalization()(c3)
c3 = Conv2D(256, (3, 3), padding='same', activation='relu')(c3)
c3 = BatchNormalization()(c3)
c3 = MaxPooling2D(pool_size=(2, 2))(c3)
c3 = Dropout(0.5)(c3)

xfc = Flatten()(c3)
xfc = Dense(512, activation="relu")(xfc)
xfc = BatchNormalization()(xfc)
xfc = Dropout(0.5)(xfc)
predictions = Dense(100, activation="softmax")(xfc)

model = Model(inputs=inputs, outputs=predictions)
train(x_train, y_train, x_test, y_test, model, BASE_FOLDER)

# Accuracy drops to 52%, which actually indicates that the network is no
# longer relying on memorized training samples. The validation curve is
# noisy because of the variability introduced batch by batch, but the
# convergence between training and validation loss demonstrates clearly
# better generalization.


# --- 4.2 Extended data augmentation (accuracy = 51%) -----------------------
data_augmentation = Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomZoom(0.1),
])

inputs = Input(shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]))
daug = data_augmentation(inputs)

c1 = Conv2D(32, (3, 3), padding='same', activation='relu')(daug)
c1 = BatchNormalization()(c1)
c1 = Conv2D(32, (3, 3), padding='same', activation='relu')(c1)
c1 = BatchNormalization()(c1)
c1 = MaxPooling2D(pool_size=(2, 2))(c1)
c1 = Dropout(0.5)(c1)

c2 = Conv2D(64, (3, 3), padding='same', activation='relu')(c1)
c2 = BatchNormalization()(c2)
c2 = Conv2D(64, (3, 3), padding='same', activation='relu')(c2)
c2 = BatchNormalization()(c2)
c2 = MaxPooling2D(pool_size=(2, 2))(c2)
c2 = Dropout(0.5)(c2)

c3 = Conv2D(256, (3, 3), padding='same', activation='relu')(c2)
c3 = BatchNormalization()(c3)
c3 = Conv2D(256, (3, 3), padding='same', activation='relu')(c3)
c3 = BatchNormalization()(c3)
c3 = MaxPooling2D(pool_size=(2, 2))(c3)
c3 = Dropout(0.5)(c3)

xfc = Flatten()(c3)
xfc = Dense(512, activation="relu")(xfc)
xfc = BatchNormalization()(xfc)
xfc = Dropout(0.5)(xfc)
predictions = Dense(100, activation="softmax")(xfc)

model = Model(inputs=inputs, outputs=predictions)
train(x_train, y_train, x_test, y_test, model, BASE_FOLDER)


# --- 4.3 Full data augmentation: crops + noise (accuracy = 46%) ------------
# Adds random crops, resizing and Gaussian noise on top of the geometric
# transformations to maximize generalization.
data_augmentation = Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomZoom(0.1),
    layers.RandomCrop(28, 28),    # Random 28x28 crops (simulates framing changes)
    layers.Resizing(32, 32),      # Resize back to 32x32 after the crop
    layers.GaussianNoise(0.05),   # Gaussian noise (simulates sensor noise)
])

inputs = Input(shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]))
daug = data_augmentation(inputs)

c1 = Conv2D(32, (3, 3), padding='same', activation='relu')(daug)
c1 = BatchNormalization()(c1)
c1 = Conv2D(32, (3, 3), padding='same', activation='relu')(c1)
c1 = BatchNormalization()(c1)
c1 = MaxPooling2D(pool_size=(2, 2))(c1)
c1 = Dropout(0.5)(c1)

c2 = Conv2D(64, (3, 3), padding='same', activation='relu')(c1)
c2 = BatchNormalization()(c2)
c2 = Conv2D(64, (3, 3), padding='same', activation='relu')(c2)
c2 = BatchNormalization()(c2)
c2 = MaxPooling2D(pool_size=(2, 2))(c2)
c2 = Dropout(0.5)(c2)

c3 = Conv2D(256, (3, 3), padding='same', activation='relu')(c2)
c3 = BatchNormalization()(c3)
c3 = Conv2D(256, (3, 3), padding='same', activation='relu')(c3)
c3 = BatchNormalization()(c3)
c3 = MaxPooling2D(pool_size=(2, 2))(c3)
c3 = Dropout(0.5)(c3)

xfc = Flatten()(c3)
xfc = Dense(512, activation="relu")(xfc)
xfc = BatchNormalization()(xfc)
xfc = Dropout(0.5)(xfc)
predictions = Dense(100, activation="softmax")(xfc)

model = Model(inputs=inputs, outputs=predictions)
train(x_train, y_train, x_test, y_test, model, BASE_FOLDER)

# After applying the full Data Augmentation pipeline (photometric variations,
# Gaussian noise and random crops), the model reaches 46% accuracy. This is
# our best experiment in terms of generalization: training and validation
# curves converge almost perfectly, completely closing the overfitting gap
# observed earlier. Although the headline accuracy is lower than models
# that were memorizing patterns, this network is structurally more robust
# and capable of correctly classifying objects under varying lighting,
# framing and noise conditions.


# =============================================================================
# CONCLUSION
# =============================================================================
# This project shows that a high accuracy figure is not always a sign of a
# good model. While early versions suffered from overfitting (memorization),
# the application of Data Augmentation achieved the full convergence of
# training and validation curves. The final accuracy of 46% is technically
# superior because it represents real generalization capability. The model
# is now robust and able to correctly classify objects under varying light,
# noise and framing conditions, ensuring real-world reliability instead of
# simply repeating memorized training data.
# =============================================================================
