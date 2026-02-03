import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds

# 1. Load the EMNIST Balanced dataset (47 classes: 0-9, A-Z, and some a-z)
(ds_train, ds_test), ds_info = tfds.load(
    'emnist/balanced',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

# 2. Preprocess: Normalize images (0-255 to 0-1) and batch them
def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(normalize_img).cache().shuffle(1000).batch(128)
ds_test = ds_test.map(normalize_img).batch(128)

# 3. Build the CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2), # Prevents overfitting
    layers.Dense(47, activation='softmax') # 47 output classes
])

# 4. Compile and Train
model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

model.fit(ds_train, epochs=10, validation_data=ds_test)

# 5. Save the model for offline use in your desktop app
model.save('handwriting_model.h5')
print("Model saved as handwriting_model.h5")