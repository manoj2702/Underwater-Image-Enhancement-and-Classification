import numpy as np 
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from numpy import argmax
from sklearn.metrics import confusion_matrix, classification_report
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import cohen_kappa_score
from tkinter import filedialog
import math


#==============================Input Data======================================
path = 'dataset/'

# categories
categories = ['Corals',  'Crabs', 'Dolphin', 'Eel', 'Jelly Fish', 'Lobster',
              'Nudibranchs', 'Octopus', 'Penguin', 'Puffers', 'Sea Rays',
              'Sea Urchins', 'Seahorse', 'Seal', 'Sharks', 'Squid', 'Starfish',
              'Turtle_Tortoise', 'Whale']

    
shape0 = []
shape1 = []

for category in categories:
    for files in os.listdir(path+category):
        shape0.append(cv2.imread(path+category+'/'+ files).shape[0])
        shape1.append(cv2.imread(path+category+'/'+ files).shape[1])
    print(category, ' => height min : ', min(shape0), 'width min : ', min(shape1))
    print(category, ' => height max : ', max(shape0), 'width max : ', max(shape1))
    shape0 = []
    shape1 = []

#===============================PRE-PROCESSING=================================
#=================================Input Image==================================
# initialize the data and labels
data = []
labels = []
imagePaths = []
HEIGHT = 65
WIDTH = 65
N_CHANNELS = 3

# grab the image paths and randomly shuffle them
for k, category in enumerate(categories):
    for f in os.listdir(path+category):
        imagePaths.append([path+category+'/'+f, k]) 

import random
random.shuffle(imagePaths)
print(imagePaths[:10])

# loop over the input images
for imagePath in imagePaths:
# load the image, resize the image to be HEIGHT * WIDTH pixels (ignoring aspect ratio) and store the image in the data list
    image = cv2.imread(imagePath[0])
    image = cv2.resize(image, (WIDTH, HEIGHT))  # .flatten()
    data.append(image)
    
    # extract the class label from the image path and update the
    # labels list
    label = imagePath[1]
    labels.append(label)
    
# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# Let's check everything is ok
fig, _ = plt.subplots(4,5)
fig.suptitle("Sample Input")
fig.patch.set_facecolor('xkcd:white')
for i in range(20):
    plt.subplot(4,5, i+1)
    plt.imshow(data[i])
    plt.axis('off')
#    plt.title(categories[labels[i]])
plt.show()

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)
# Preprocess class labels
test = np_utils.to_categorical(labels, 19)
trainY = np_utils.to_categorical(trainY, 19)
y_test = np_utils.to_categorical(testY, 19)
print(trainX.shape)
print(testX.shape)
print(trainY.shape)
print(testY.shape)

#================================Classification================================
'''Convolutional Neural Network'''

print('Convolutional Neural Network')
model = Sequential()

model.add(Convolution2D(32, (2, 2), activation='relu', input_shape=(HEIGHT, WIDTH, N_CHANNELS)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(19, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

history = model.fit(data, test, batch_size=32, epochs=25, verbose=1)
cnn=model.evaluate(testX,y_test)[1]*100
print("Accuracy of the CNN is:",model.evaluate(testX,y_test)[1]*100, "%")
#history=model.history.history
#Plotting the accuracy
train_loss = history.history['loss']
train_acc = history.history['acc']

    
# Performance graph
plt.figure()
plt.plot(train_loss, label='Loss')
plt.plot(train_acc, label='Accuracy')
plt.title('Performance Plot')
plt.legend()
plt.show()
    
#=============================Analytic Results=================================
pred = model.predict(testX)
predictions = argmax(pred, axis=1) 
print('Classification Report')
cr=classification_report(testY, predictions,target_names=categories)
print(cr)
print()
print('Cohen Kappa Coefficient')
cks=cohen_kappa_score(testY, predictions)
print('Cohen Kappa Coefficient:',cks)
print()
print('Confusion Matrix')
cm = confusion_matrix(testY, predictions)
print(cm)
#Confusion Matrix Plot
plt.figure()
plot_confusion_matrix(cm,figsize=(15,15), class_names = categories,
                      show_normed = True);

plt.title( "Model confusion matrix")
plt.style.use("ggplot")
plt.show()

#================================Comparison====================================
a=80
vals=[a, cnn]
inds=range(len(vals))
labels=["A", "CNN"]
fig,ax = plt.subplots()
rects = ax.bar(inds, vals)
ax.set_xticks([ind for ind in inds])
ax.set_xticklabels(labels)
plt.show()
#=================================Prediction===================================
test_data=[]
file = filedialog.askopenfilename()
head_tail = os.path.split(file)
fileNo=head_tail[1].split('.')
Image = cv2.imread(head_tail[0]+'/'+fileNo[0]+'.jpg')
img=cv2.resize(Image,(512,512))
#================================white Balance=================================
def show(final):
    cv2.imshow('White Balanced', final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def white_balance_loops(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    for x in range(result.shape[0]):
        for y in range(result.shape[1]):
            l, a, b = result[x, y, :]
            # fix for CV correction
            l *= 100 / 255.0
            result[x, y, 1] = a - ((avg_a - 128) * (l / 100.0) * 1.1)
            result[x, y, 2] = b - ((avg_b - 128) * (l / 100.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

final = np.hstack((img, white_balance_loops(img)))
show(final)
#r,g,b balance
r, g, b = cv2.split(img)
r_avg = cv2.mean(r)[0]
g_avg = cv2.mean(g)[0]
b_avg = cv2.mean(b)[0]
 
# Find the gain of each channel
k = (r_avg + g_avg + b_avg) / 3
kr = k / r_avg
kg = k / g_avg
kb = k / b_avg
 
r = cv2.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
g = cv2.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
b = cv2.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)
 
balance_img = cv2.merge([b, g, r])
cv2.imshow('Balanced', balance_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
plt.imshow(balance_img)
plt.show()
#===============================History Equalization===========================

import numpy as np
import matplotlib
from skimage import img_as_float
from skimage import exposure


matplotlib.rcParams['font.size'] = 8


def plot_img_and_hist(image, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram.

    """
    image = img_as_float(image)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf


# Contrast stretching
p2, p98 = np.percentile(img, (2, 98))
img_con = exposure.rescale_intensity(img, in_range=(p2, p98))

# Equalization
img_eq = exposure.equalize_hist(img)

# Adaptive Equalization
img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)

# Display results
fig = plt.figure(figsize=(8, 5))
axes = np.zeros((2, 4), dtype=np.object)
axes[0, 0] = fig.add_subplot(2, 4, 1)
for i in range(1, 4):
    axes[0, i] = fig.add_subplot(2, 4, 1+i, sharex=axes[0,0], sharey=axes[0,0])
for i in range(0, 4):
    axes[1, i] = fig.add_subplot(2, 4, 5+i)

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img, axes[:, 0])
ax_img.set_title('Low contrast image')

y_min, y_max = ax_hist.get_ylim()
ax_hist.set_ylabel('Number of pixels')
ax_hist.set_yticks(np.linspace(0, y_max, 5))

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_con, axes[:, 1])
ax_img.set_title('Contrast stretching')

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_eq, axes[:, 2])
ax_img.set_title('Histogram equalization')

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_adapteq, axes[:, 3])
ax_img.set_title('Adaptive equalization')

ax_cdf.set_ylabel('Fraction of total intensity')
ax_cdf.set_yticks(np.linspace(0, 1, 5))

# prevent overlap of y-axis labels
fig.tight_layout()
plt.show()

cv2.imshow('Low contrast image', img)
cv2.imshow('Contrast stretching', img_con)
cv2.imshow('Histogram equalization', img_eq)
cv2.imshow('Adaptive equalization', img_adapteq)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.title('Low contrast image')
plt.imshow(img)
plt.show()
plt.title('Contrast stretching')
plt.imshow(img_con)
plt.show()
plt.title('Histogram equalization')
plt.imshow(img_eq)
plt.show()
plt.title('Adaptive equalization')
plt.imshow(img_adapteq)
plt.show()

#================================GAMMA CORRECTION==============================
# convert img to gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# compute gamma = log(mid*255)/log(mean)
mid = 0.5
mean = np.mean(gray)
gamma = math.log(mid*255)/math.log(mean)
print(gamma)

# do gamma correction
img_gamma1 = np.power(img, gamma).clip(0,255).astype(np.uint8)

# convert img to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hue, sat, val = cv2.split(hsv)

# compute gamma = log(mid*255)/log(mean)
mid = 1
mean = np.mean(val)
gamma = math.log(mid*255)/math.log(mean)
print(gamma)

# do gamma correction on value channel
val_gamma = np.power(val, gamma).clip(0,255).astype(np.uint8)

# combine new value channel with original hue and sat channels
hsv_gamma = cv2.merge([hue, sat, val_gamma])
img_gamma2 = cv2.cvtColor(hsv_gamma, cv2.COLOR_HSV2BGR)

cv2.imshow('GAMMA = 0.5', img_gamma1)
cv2.imshow('GAMMA = 1', img_gamma2)
cv2.waitKey(0)
cv2.destroyAllWindows()


plt.imshow(img_gamma1)
plt.show()
plt.imshow(img_gamma2)
plt.show()

test_image = cv2.resize(Image, (WIDTH, HEIGHT))
test_data.append(test_image)

# scale the raw pixel intensities to the range [0, 1]
test_data = np.array(test_image, dtype="float") / 255.0
test_data=test_data.reshape([-1,65, 65, 3])
pred = model.predict(test_data)
predictions = argmax(pred, axis=1) # return to label
print ('Prediction : '+categories[predictions[0]])

#Imersing into the plot
fig = plt.figure()
fig.patch.set_facecolor('xkcd:white')
plt.imshow(Image)