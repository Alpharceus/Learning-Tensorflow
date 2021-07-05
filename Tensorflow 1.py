import tensorflow as tf
print("\n tensorflow version:"), tf._version_)
tensorflow version: 2.4.1
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

import matplotlib.pyplot as plt
import numpy as np


train_images.shape
(60000, 28 ,28)
 
 test_images.shape

#Model Create

model = tf.keras.models.Sequential(
[
    tf.keras.layers.Flatten(input_shape(28,28)),
    tf.keras.layers.Dense(16 activation="relu"))
    tf.keras.layers.Dense(10 activation="softmax"))
    
    
    
    
    
    
 #model compelete
 model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
 
 #training 
 H = model.fit(train_images, train_labels, validation_data=(test_images, test_labels)=10)
 imoprt pandas as pd
 frame =pd.Dataframe(H.history)
 frame.head()
 acc_plot= frame.plot(y="accuracy", title="Acc. vs Epoch", legends =False)
 #acc_plot.set(xlabels="Epochs", ylabels="Accuracy")
 pit.show()
 
 #Evaluate
 lest_loss, test_accuracy =model.evaluate(test_images, test_labels)
 
 lest_loss, test_accuracy 
 
 # lets get a model prediction on randomly selected images
 
 num_test_images = test_images.shape[0]
 random_index= np.random.choice(num_test_image, 4)
 randon_test_images = test_images(random_index)
 random_test_labels = test_labels[random_index]
 predictions = model.predict(random_test_images)
 
 fig, axes =pit.subplots(4,2, figsize(16,12))
 fig.subplots_adjust(hspace=0.4,wspace=-0.2)
 for i, (prediction,images, label)in enumerate (zip,predictions, random_test_images, random test_labels)):
    axes[i,0].imshow(np.squeeze(image))
    axes[i,0].get_xaxis().set_visible(False)
    axes[i,0].get_yaxis().set_visible(False)
    axes[i,0].text(10., -1.5, f"Digit {label}")
    axes[i,1].plot(prediction)
    axes[i,1].set_xticks(np.arrage(len(prediction)))
    axes[i,1].set_title(f"
 
 
 