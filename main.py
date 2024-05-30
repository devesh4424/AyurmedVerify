import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import numpy as np

IMAGE_SIZE=256
BATCH_SIZE=20
CHANNELS=3
EPOCHS=50

images_dataset=tf.keras.preprocessing.image_dataset_from_directory(
 "D:\Datasets\Medicinal_Leaf_dataset",
 shuffle=True,
 image_size=(IMAGE_SIZE,IMAGE_SIZE),
 batch_size=BATCH_SIZE,
)

class_names=images_dataset.class_names
class_names

#exploring the dataset
for image_batch, label_batch in images_dataset.take(1):
    print(image_batch.shape)
    print(label_batch.numpy())

#visualize the first image in that batch
for image_batch, label_batch in images_dataset.take(1):
    plt.imshow(image_batch[0].numpy().astype('uint8'))
    plt.title(class_names[label_batch[0]])
    plt.axis('off')

#visualize the first image in that batch
plt.figure(figsize=(18,18))
for image_batch, label_batch in images_dataset.take(1):
    for i in range (12):
        ax=plt.subplot(3,4, i+1)
        plt.imshow(image_batch[i].numpy().astype('uint8'))
        plt.title(class_names[label_batch[i]])
        plt.axis('off')

len(images_dataset)

def get_dataset_partitions_tf(ds,train_split=0.8, val_split=0.1, test_split=0.1,shuffle=True, shuffle_size=10000):
    ds_size=len(ds)
    if shuffle:
        ds=ds.shuffle(shuffle_size,seed=12)

    train_size= int(train_split* ds_size)
    val_size=int(val_split* ds_size)

    train_ds=ds.take(train_size)

    val_ds=ds.skip(train_size).take(val_size)

    test_ds=ds.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds

#train test split
train_ds, val_ds, test_ds=get_dataset_partitions_tf(images_dataset)

train_ds=train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
val_ds=val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_ds=test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

resize_and_rescale=tf.keras.Sequential([

    layers.experimental.preprocessing.Resizing(IMAGE_SIZE,IMAGE_SIZE),
    layers.experimental.preprocessing.Rescaling(1.0/255)
])

data_augmentation=tf.keras.Sequential([

    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2)
])

input_shape=(BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,CHANNELS)
n_classes=80
model=models.Sequential([
    resize_and_rescale,
    data_augmentation,
    # layers.Conv2D(32,(3,3),activation='relu',input_shape=input_shape),
    # layers.MaxPooling2D((2,2)),
    # layers.Conv2D(64,(3,3),activation='relu',input_shape=input_shape),
    # layers.MaxPooling2D((2,2)),
    # layers.Conv2D(64,kernel_size=(3,3),activation='relu'),
    # layers.MaxPooling2D((2,2)),
    # layers.Conv2D(128,(3,3),activation='relu'),
    # layers.MaxPooling2D((2,2)),
    # layers.Conv2D(128,(3,3),activation='relu'),
    # layers.MaxPooling2D((2,2)),
    # layers.Flatten(),
    # tf.keras.layers.Dropout(0.5),
    # layers.Dense(512,activation='tanh'),
    # layers.Dense(n_classes,activation='softmax'),
    Z1 = tf.keras.layers.Conv2D(8, (4, 4), strides=(1, 1), padding="same")(input_img)
A1 = tf.keras.layers.ReLU()(Z1)
P1 = tf.keras.layers.MaxPooling2D(pool_size=(8, 8), strides=(4, 4), padding="same")(A1)

# Second convolutional layer
Z2 = tf.keras.layers.Conv2D(filters=16,kernel_size=(2, 2), strides=(1, 1), padding="same")(P1)
A2 = tf.keras.layers.ReLU()(Z2)
P2 = tf.keras.layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding="same")(A2)

# Third convolutional layer
Z3 = tf.keras.layers.Conv2D(32, (4, 4), strides=(1, 1), padding="same")(P2)
A3 = tf.keras.layers.ReLU()(Z3)
P3 = tf.keras.layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding="valid")(A3)

# Skip Connection Z1
S1 = tf.keras.layers.Conv2D(16,kernel_size=(1,1),strides=(1,1),padding="same")(Z1)
S1_pooled = tf.keras.layers.MaxPool2D(pool_size=(56,56), strides=(8,8),padding = "same")

# Fourth convolutional layer
Z4 = tf.keras.layers.Conv2D(filters=16,kernel_size=(2, 2), strides=(1, 1), padding="same")(S1)
A4 = tf.keras.layers.ReLU()(Z4)
P4 = tf.keras.layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding="same")(A4)

F = tf.keras.layers.Flatten()(P4)
D1 = tf.keras.layers.Dense(units = 16, activation="tanh")(F)
outputs = tf.keras.layers.Dense(units=3,activation="softmax")(D1)
])
model.build(input_shape=input_shape)

#summary of model
model.summary()

#compile the model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

history=model.fit(
     train_ds,
     epochs=EPOCHS,
     batch_size=BATCH_SIZE,
     verbose = 1 ,
     validation_data=val_ds
)

#evaluate the model
scores=model.evaluate(test_ds)
print(f' Accuracy: {scores[1]*100:.2f}%')


# Save the model for later use
model.save(r"D:\Datasets\plant_classifier_model.h5")

# Load the model
loaded_model = tf.keras.models.load_model(r"D:\Datasets\plant_classifier_model.h5")

# Visualize the training and validation loss/accuracy curves
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

# Generate confusion matrix for the training set
y_train_pred = np.argmax(model.predict(train_ds), axis=1)
cm_train = confusion_matrix(train_ds.labels, y_train_pred)
sns.heatmap(cm_train, annot=True, fmt='d')
plt.show()

# Generate confusion matrix for the validation set
y_val_pred = np.argmax(model.predict(val_ds), axis=1)
cm_val = confusion_matrix(val_ds.labels, y_val_pred)
sns.heatmap(cm_val, annot=True, fmt='d')
plt.show()
