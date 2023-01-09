import numpy
import pandas
import os
import ntpath
import cv2
import random

# MatplotLib:
from matplotlib import pyplot as pyplot
from matplotlib import image as matimg

# SkLearn:
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Keras:
from keras.applications import ResNet50
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import Adam


# Creating a Dataset from dataset_log.csv
# Store the Data:
data_directory = "D:\Coding\Python\Projects\Self-Driving Car"
columns = ["center", "left", "right", "steering", "throttle", "reverse", "speed"]
dataset = pandas.read_csv(os.path.join(data_directory, "D:\Coding\Python\Projects\Self-Driving Car\Dataset\driving_log.csv"), names=columns)
pandas.set_option("display.max_colwidth", None)

print(dataset.head(10))

# Removing Prefixes from the path of Frames:
def path_leaf(path):
        head, tail = ntpath.split(path)
        return tail

dataset["center"] = dataset["center"].apply(path_leaf)
dataset["left"] = dataset["left"].apply(path_leaf)
dataset["right"] = dataset["right"].apply(path_leaf)

print("--------------------------------------------------------------------------------------")
print(dataset.head(10))


# Analyzing the Data:
# Plotting the Steering Angles Data:
num_bins = 25
samples_per_bin = 200
hist, bins = numpy.histogram(dataset["steering"], num_bins)
center = bins[:-1] + bins[1:] * 0.5     # Centering the Bins to 0

pyplot.bar(center, hist, width=0.05)
pyplot.plot((numpy.min(dataset["steering"]), numpy.max(dataset["steering"])), (samples_per_bin, samples_per_bin))
pyplot.show()

print("----------------------------------------------------------------------------------------")
print(f"Total Data => {len(dataset)} ")

# Making a List of Indices to Remove:
remove_list = []

for i in range(num_bins):
        temp_list = []
        for j in range(len(dataset["steering"])):
                steering_angle = dataset["steering"][j]

                if steering_angle >= bins[i] and steering_angle <= bins[i + 1]:
                        temp_list.append(j)
        
        temp_list = shuffle(temp_list)
        temp_list = temp_list[samples_per_bin:]
        remove_list.extend(temp_list)

# Remove extras from List:
print("\n----------------------------------------------------------------------------------")
dataset.drop(dataset.index[remove_list], inplace=True)
print(f"Removed => {len(remove_list)}")
print(f"Remaining => {len(dataset)}")

# Plotting:
hist, _ = numpy.histogram(dataset["steering"], (num_bins))
pyplot.bar(center, hist, width=0.05)
pyplot.plot((numpy.min(dataset["steering"]), numpy.max(dataset["steering"])), (samples_per_bin, samples_per_bin))
pyplot.show()


# Loading up all the Frames and Steering Angles into a Numpy Array:
def load_img_steer(data_directory, df):
        image_paths = []
        steering_angles = []

        for i in range(len(dataset)):
                indexed_data = dataset.iloc[i]
                center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
                image_paths.append(os.path.join(data_directory, center.strip()))
                steering_angles.append(float(indexed_data[3]))
        
        numpy_image_path = numpy.asarray(image_paths)
        numpy_steering_angle = numpy.asarray(steering_angles)

        return numpy_image_path, numpy_steering_angle


numpy_image_path, numpy_steering_angle = load_img_steer(data_directory + "\Dataset\IMG", dataset)


# Splitting Data into Testing and Training:
X_Train, X_Valid, Y_Train, Y_Valid = train_test_split(numpy_image_path, numpy_steering_angle, test_size=0.2, random_state=0)

# Checking if the Data is Valid:
print(f"Training Samples => {len(X_Train)}")
print(f"Valid Samples => {len(X_Valid)}")

fig, axes = pyplot.subplots(1, 2, figsize=(12, 4))
axes[0].hist(Y_Train, bins=num_bins, width=0.05, color="#4A148C")
axes[0].set_title("Training Set: ")
axes[1].hist(Y_Valid, bins=num_bins, width=0.05, color="#880E4F")
axes[1].set_title("Validation Set: ")
pyplot.show()


# Pre-Processing the Images:
def image_preprocess(img):
        # Takes in path, returns a pre-processed Image
        img = matimg.imread(img)

        # Cropping Image to remove unneccesary features:
        img = img[60:135,  :, : ]

        # Change to YUV Image:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

        # Gaussian Blur:
        img = cv2.GaussianBlur(img, (3, 3), 0)

        # Decreasing Size for Easier Processing:
        img = cv2.resize(img, (100, 100))

        # Normalize the Values:
        img = img / 255
        return img


# Getting any Image:
sample_image = numpy_image_path[100]
original_image = matimg.imread(sample_image)
preprocessed_image = image_preprocess(sample_image)

# Comparing the Original and Pre-Processed Images:
fig, axes = pyplot.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()
axes[0].imshow(original_image)
axes[0].set_title("Original Image: ")
axes[1].imshow(preprocessed_image)
axes[1].set_title("Preprocessed Image: ")

pyplot.show()

X_Train = numpy.array(list(map(image_preprocess, X_Train)))
X_Valid = numpy.array(list(map(image_preprocess, X_Valid)))

pyplot.imshow(X_Train[random.randint(0, len(X_Train) - 1)])
pyplot.axis("off")
print(X_Train.shape)
pyplot.show()


# Keras: Loading up the ResNet50 Model:
resnet_model = ResNet50(weights="imagenet", include_top=False, input_shape=(100, 100, 3))

# Freeze the Layers except the last 4 layers:
for layer in resnet_model.layers[ :-4]:
        layer.trainable = False

print("\n----------------------------------------------------------------------------------")
for layer in resnet_model.layers:
        print(layer, layer.trainable)


def nvidia_model():
        model = Sequential()
        model.add(resnet_model)

        model.add(Dropout(0.5))

        model.add(Flatten())

        model.add(Dense(100, activation="elu"))
        model.add(Dropout(0.5))

        model.add(Dense(50, activation="elu"))
        model.add(Dropout(0.5))

        model.add(Dense(10, activation="elu"))
        model.add(Dropout(0.5))

        model.add(Dense(1))

        optimizer = Adam(lr=1e-3)
        model.compile(loss="mse", optimizer=optimizer, metrics=["accuracy"])
        return model


model_object = nvidia_model()

print("\n----------------------------------------------------------------------------------")
print(model_object.summary())


# Training the Model for 25 Epochs with batch size 128:
history = model_object.fit(X_Train, Y_Train, epochs=25, validation_data=(X_Valid, Y_Valid), batch_size=128, verbose=1, shuffle=1)

pyplot.plot(history.history["loss"])
pyplot.plot(history.history["val_loss"])
pyplot.legend(["training", "validation"])
pyplot.title("Loss: ")
pyplot.xlabel("Epoch: ")
pyplot.show()
