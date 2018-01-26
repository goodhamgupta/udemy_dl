from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()

# First convolutional layer
model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

# Max Pooling
model.add(MaxPooling2D(pool_size=(2, 2))) # 2,2 specifies the stride

# Flatten all feature maps
model.add(Flatten())

# Fully connected layer
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1), activation='sigmoid')

# Compile the CNN
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Fitting the CNN to images

# Preprocessing result

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
test_datagen = ImageDataGenerator(rescale=1./2255)
test_generator = train_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
