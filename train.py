import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#from tensorflow.keras.applications import MobileNetV2
#from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

from tensorflow.keras import regularizers
from augmentation import augmentImages


""" import ssl

# Disable SSL verification globally
ssl._create_default_https_context = ssl._create_unverified_context """

AUGMENT = False

BASE_DIR = 'data/'
EFFECTIVE_DIR = 'originalData'
names = ["breasts", "butterfliedDrumsticks", "drumsticks", "wholeLeg", "wings"]

t_size_x = 224
t_size_y = 224

# Function to delete a directory if it exists
def delete_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
        print(f"Deleted {directory}")

def show(batch, pred_labels=None):
    plt.figure(figsize=(10,10))
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(batch[0][i], cmap=plt.cm.binary)
        # The CIFAR labels happen to be arrays, 
        # which is why you need the extra index
        lbl = names[int(batch[1][i])]
        if pred_labels is not None:
            lbl += "/ Pred:" + names[int(pred_labels[i])]
        plt.xlabel(lbl)
    plt.show()


# Delete train, val, and test directories
delete_directory(BASE_DIR + 'train/')
delete_directory(BASE_DIR + 'val/')
delete_directory(BASE_DIR + 'test/')

# Reorganize the folder structure:
if not ( os.path.isdir(BASE_DIR + 'train/') or os.path.isdir(BASE_DIR + 'test/') or os.path.isdir(BASE_DIR + 'val/') ):
    for name in names:
        os.makedirs(BASE_DIR + 'train/' + name)
        os.makedirs(BASE_DIR + 'val/' + name)
        os.makedirs(BASE_DIR + 'test/' + name)

# Move the image files
orig_folders = [f"{EFFECTIVE_DIR}/breasts/", f"{EFFECTIVE_DIR}/butterfliedDrumsticks/", f"{EFFECTIVE_DIR}/drumsticks/", f"{EFFECTIVE_DIR}/wholeLeg/", f"{EFFECTIVE_DIR}/wings/"]
for folder_idx, folder in enumerate(orig_folders):
    files = os.listdir(BASE_DIR + folder)
    number_of_images = len([name for name in files])
    n_train = int((number_of_images * 0.8) + 0.5)
    n_valid = int((number_of_images * 0.1) + 0.5)
    n_test = number_of_images - n_train - n_valid
    for idx, file in enumerate(files):
        file_name = BASE_DIR + folder + file
        if idx < n_train:
            shutil.copy(file_name, BASE_DIR + "train/" + names[folder_idx])
        elif idx < n_train + n_valid:
            shutil.copy(file_name, BASE_DIR + "val/" + names[folder_idx])
        else:
            shutil.copy(file_name, BASE_DIR + "test/" + names[folder_idx])

if AUGMENT: augmentImages()

# Preprocess the data
train_gen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)
valid_gen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)
test_gen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)

train_batches = train_gen.flow_from_directory(
    f'{BASE_DIR}/train',
    target_size=(t_size_x, t_size_y),  # Resnet50 expects 224x224 input
    class_mode='sparse',
    batch_size=16,
    shuffle=True,
    color_mode="rgb",
    classes=names
)

val_batches = valid_gen.flow_from_directory(
    f'{BASE_DIR}/val',
    target_size=(t_size_x, t_size_y),
    class_mode='sparse',
    batch_size=16,
    shuffle=False,
    color_mode="rgb",
    classes=names
)

test_batches = test_gen.flow_from_directory(
    f'{BASE_DIR}/test',
    target_size=(t_size_x, t_size_y),
    class_mode='sparse',
    batch_size=16,
    shuffle=False,
    color_mode="rgb",
    classes=names
)

# Load pre-trained MobileNetV2 as the base model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(t_size_x, t_size_y, 3))
base_model.trainable = False # Freeze the base model

print(base_model.summary())


# Build the new model
inputs = tf.keras.Input(shape=(t_size_x, t_size_y, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(len(names), activation='softmax')(x)
model = models.Model(inputs, outputs)

outputs = layers.Dense(len(names), activation='softmax')(x)  # Output layer with softmax for classification

# Create the model
model = keras.Model(inputs, outputs)

#print(model.summary())

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# callbacks
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    verbose=1
)
reduce_lr_on_plateau = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=3,
    verbose=1
)

# Training callbacks
callbacks = [
    early_stopping,
    reduce_lr_on_plateau
]

# Train the model
epochs = 30
history = model.fit(train_batches, validation_data=val_batches, epochs=epochs, verbose=2, callbacks=callbacks)

# Unfreeze some layers of the base model for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:50]:  # Freeze the first 50 layers
    layer.trainable = False

# Compile and fine-tune the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Fine-tune
fine_tune_epochs = 10
total_epochs = epochs + fine_tune_epochs

history_fine = model.fit(train_batches, validation_data=val_batches, epochs=total_epochs, initial_epoch=history.epoch[-1], verbose=2)

# Save the model
model.save("models/MODEL-Q.keras")

# Evaluate the model
test_loss, test_acc = model.evaluate(test_batches, verbose=2)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# make some predictions
predictions = model.predict(test_batches, verbose=1)
#predictions = tf.nn.softmax(predictions)
predicted_labels = np.argmax(predictions, axis=1)

# Example for the first batch of test data
batch_data = next(test_batches) # Get the next batch from test_batches

print("True Labels for the batch: ", batch_data[1][:4])  # Print true labels for the first 4 images
print("Predicted Labels for the batch: ", predicted_labels[:4])  # Print predicted labels for the first 4 images

# Show the first 4 images in the batch along with predictions
show(batch_data, predicted_labels[:4])
