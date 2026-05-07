# Galaxy Redshift Prediction using Images

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255
)

# Training dataset
train_data = train_datagen.flow_from_directory(
    "galaxy_images/train",
    target_size=(128,128),
    batch_size=32,
    class_mode='raw'
)

# Testing dataset
test_data = train_datagen.flow_from_directory(
    "galaxy_images/test",
    target_size=(128,128),
    batch_size=32,
    class_mode='raw'
)

# CNN Model
model = Sequential()

# First convolution layer
model.add(
    Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(128,128,3)
    )
)

model.add(MaxPooling2D(2,2))

# Second convolution layer
model.add(
    Conv2D(
        64,
        (3,3),
        activation='relu'
    )
)

model.add(MaxPooling2D(2,2))

# Flatten layer
model.add(Flatten())

# Dense layer
model.add(Dense(128, activation='relu'))

# Output layer
model.add(Dense(1))

# Compile model
model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mae']
)

# Train model
model.fit(
    train_data,
    epochs=10
)

# Evaluate
loss, mae = model.evaluate(test_data)

print("Test MAE:", mae)
