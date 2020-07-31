**Import libraries:**

     import numpy as np
     import pandas as pd
     from tensorflow import keras
     from tensorflow.keras import layers
     import matplotlib.pyplot as plt
     from sklearn.metrics import confusion_matrix
     from keras.utils import to_categorical
     
     from keras.models import Sequential
     from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
     from keras.optimizers import adam
     from keras.preprocessing.image import ImageDataGenerator
     from keras.callbacks import ReduceLROnPlateau
     
 ** Load the training and test files **
 
    train_df = pd.read_csv("Downloads/digit-recognizer/train.csv")
    test_df = pd.read_csv("Downloads/digit-recognizer/test.csv")
    
    X = train_df.drop('label', axis=1)
    y = train_df['label']
    
 **Normalization**
   
   We perform the grayscale normalization to reduce the effect of illumination's differences. The values of the pixels are 0 to 255, 
   we're normalized them to 0 to 1.
    
     X = X.astype("float32")/255
     test_df = test_df.astype("float32")/255
     
 **Reshape the data**
 
  Train and test images are of size 28 x 28 and are stock into pandas DataFrame as a 1D vector with 784 values. 
  
  For Keras we will reshape them to 28x28x1 3D matrics. It is a MINST dataset has greyscale images, so there is only 1 channel, if it is an RGB image, then there are 3 channels.     
    
    X = X.values.reshape(-1, 28, 28, 1) 
    test_df = test_df.values.reshape(-1, 28, 28, 1)
    
   **Label Encoding** 
   
   The labels are 10 numbers, 0 to 9. We encode the labels to one hot vectors (_e.g., 2 -> [0, 0, 1, 0, 0, 0, 0, 0, 0]_) 
   
    y = to_categorical(y, num_classes = 10)
    
  **Split the training set into training and validation set:**  
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
  **Create the model**  
  This CNN architechture is _In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out_
  
      model = Sequential()
      
      model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                       activation ='relu', input_shape = (28,28,1)))
      model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                       activation ='relu'))
      model.add(MaxPool2D(pool_size=(2,2)))
      model.add(Dropout(0.25))
      
      model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                       activation ='relu', input_shape = (28,28,1)))
      model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                       activation ='relu'))
      model.add(MaxPool2D(pool_size=(2,2)))
      model.add(Dropout(0.25))
      
      model.add(Flatten())
      model.add(Dense(256, activation = "relu"))
      model.add(Dropout(0.5))
      model.add(Dense(10, activation = "softmax"))
  
    
    
  **Set the optimizers**
  
  Once the layers are added to the model, we need to set up a score function, a loss function and an optimization algorithm.
  
  We define the loss function to measure how poorly our model performs on images with known labels. It is the error rate between the oberved labels and the predicted ones. We use a specific form for categorical classifications (>2 classes, as we have 10 classes) called the "categorical_crossentropy".
    
  The most important function is the optimizer. This function will iteratively improve parameters (filters kernel values, weights and bias of neurons ...) in order to minimise the loss.
  
  Adam optimizer with default values is used in here.
     
     adam = adam()
     
     model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
     
  **Train the model**
  
    epochs = 30 
    batch_size = 70
    
    history = model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, validation_data = (X_test, y_test), verbose = 2)
    
    
  **Evaluate the model**
  
    # Plot the loss and accuracy curves for training and validation 
    fig, ax = plt.subplots(2,1)
    ax[0].plot(history.history['loss'], color='b', label="Training loss")
    ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
    legend = ax[0].legend(loc='best', shadow=True)
    
    ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
    ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
    
  Our model has 99.3% accuracy, which is pretty good.
  
  Now it's time to predict the test dataset. 
    
    Y_pred = model.predict(test_df)
  
       
     
    
 