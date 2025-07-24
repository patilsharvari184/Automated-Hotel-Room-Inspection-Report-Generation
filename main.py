import numpy as np # type: ignore
import os
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import tensorflow as tf # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.models import Sequential, load_model # type: ignore
from keras.layers import Conv2D, MaxPool2D, Flatten,GlobalAveragePooling2D,Dense,Dropout,BatchNormalization # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.applications.vgg16 import preprocess_input # type: ignore
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau # type: ignore
from tensorflow.keras.layers import BatchNormalization, Dropout # type: ignore
from keras.layers import Input, Lambda, Dense, Flatten # type: ignore
from tensorflow.keras.applications import VGG16 # type: ignore
from keras import regularizers # type: ignore
from tensorflow.keras.optimizers import Adam,RMSprop,SGD,Adamax,AdamW # type: ignore
from IPython import get_ipython
from IPython.display import display
import datetime
import os
import random
import matplotlib.image as mpimg # type: ignore
from reportlab.lib.pagesizes import letter # type: ignore
from reportlab.pdfgen import canvas # type: ignore
from reportlab.lib.utils import ImageReader # type: ignore

train_dir = "E:/Benchmark/Automated Inspection Report/hotel_dataset/Dataset/train" #passing the path with training images
validation_dir = "E:/Benchmark/Automated Inspection Report/hotel_dataset/Dataset/val"   #passing the path with validation images

#preprocessing the images
IMG_SIZE= 500 #image size

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # No rescale
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # No rescale
    validation_split=0.2
)

# Load the images
train_generator = train_datagen.flow_from_directory(directory = train_dir,
                                                  target_size = (IMG_SIZE,IMG_SIZE),
                                                   color_mode = "rgb",
                                                   class_mode = "categorical",
                                                   batch_size = 32,
                                                   subset = "training")
validation_generator = validation_datagen.flow_from_directory(directory = validation_dir,
                                                   target_size = (IMG_SIZE,IMG_SIZE),
                                                   color_mode = "rgb",
                                                   class_mode = "categorical",
                                                   batch_size = 32,
                                                   subset = "validation")

# Define the VGG16 model
#model = tf.keras.Sequential()
#model.add(VGG16(include_top = False,weights = 'imagenet',input_shape= (IMG_SIZE,IMG_SIZE,3)))
#model.add(Flatten())
#model.add(Dense(2, activation='softmax', kernel_regularizer=regularizers.l2(0.01)))
#model.add(Dropout(0.5))  # Increase dropout rate

# Define the model
base_model = VGG16(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))# Load the VGG16 model
base_model.trainable = False  # Freeze VGG16 layers

model = tf.keras.Sequential([
    base_model,
    GlobalAveragePooling2D(),  # Better than Flatten()
    Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)),Dropout(0.5),  # Dropout before final layer
    Dense(2, activation='softmax')  # Output layer for 2 classes
])

# Define the optimizer
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau # type: ignore # Import ModelCheckpoint

loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
optimizer = Adam(learning_rate = 0.00001) # learning rate
#optimizer = AdamW(learning_rate=0.00001, weight_decay=1e-4)
model.compile(optimizer=optimizer , loss='binary_crossentropy', metrics=['accuracy'])# Compile the model

# Define Learning Rate Scheduling
#lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, verbose=1)
#lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1)

# Define EarlyStopping
early_stop = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=7,  # Stop if val_loss doesn’t improve for 3 consecutive epochs
    restore_best_weights=True  # Restore the best weights after stopping
)

# Assign the VGG16 model to base_model
base_model = model.layers[0]  # VGG16 is the first layer in your Sequential model

#for layer in base_model.layers[-10:]:  # Unfreeze last 30 layers
#    layer.trainable = False
#for layer in model.layers:
#    layer.trainable = False
#for layer in model.layers[-4:]:
#    layer.trainable = True  # Unfreeze the last few layers

# Define ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    'my_model_epoch_{epoch:02d}.keras',
    save_best_only=False,
    save_weights_only=False,
    monitor='val_loss',
    verbose=1,
    save_freq='epoch'  # Save every epoch
)

model.summary()

# Train the model
epochs = 40
batch_size = 64

history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    steps_per_epoch=len(train_generator),
    validation_steps=len(validation_generator),
    callbacks=[early_stop]
)

# Plot the training and validation accuracy and loss
fig , ax = plt.subplots(1,2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
fig.set_size_inches(12,4)

ax[0].plot(history.history['accuracy'])
ax[0].plot(history.history['val_accuracy'])
ax[0].set_title('Training Accuracy vs Validation Accuracy')
ax[0].set_ylabel('Accuracy')
ax[0].set_xlabel('Epoch')
ax[0].legend(['Train', 'Validation'], loc='upper left')

ax[1].plot(history.history['loss'])
ax[1].plot(history.history['val_loss'])
ax[1].set_title('Training Loss vs Validation Loss')
ax[1].set_ylabel('Loss')
ax[1].set_xlabel('Epoch')
ax[1].legend(['Train', 'Validation'], loc='upper left')

plt.show()

# Evaluate the model
train_loss, train_acc = model.evaluate(train_generator)
test_loss, test_acc   = model.evaluate(validation_generator)
print("final train accuracy = {:.2f} , validation accuracy = {:.2f}".format(train_acc*100, test_acc*100))

# Save the model
model.save('my_model22.h5')

# Download the model
#files.download('my_model3.h5')
'''
# REPORT GENERATION

# Load Model
model = load_model('E:/Benchmark/Automated Inspection Report/model.h5')

def generate_inspection_id():
    """Generate a unique inspection ID (A1, A2, etc.)."""
    return f"A{random.randint(100, 999)}"

def predict_image(image_path):
    """Predicts cleanliness of a room image and generates a PDF report."""
    try:
        # Load and preprocess image
        img = image.load_img(image_path, target_size=(150, 150))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        img_data = preprocess_input(x)

        # Makes Prediction
        classes = model.predict(img_data)
        clean_prob = classes[0][0]
        messy_prob = classes[0][1]

        # Determine Classification
        if clean_prob > messy_prob:
            prediction = "Clean"
            confidence = clean_prob * 100
        else:
            prediction = "Messy"
            confidence = messy_prob * 100

        # Display Image
        img = mpimg.imread(image_path)
        plt.imshow(img)
        plt.axis('off')
        plt.show()

        # Generate PDF Report
        report_data = {
            "inspection_id": generate_inspection_id(),
            "date": datetime.datetime.now().strftime("%d/%m/%Y"),
            "hotel_name": "Sample Hotel",
            "room_area": "Room 302",
            "inspection_type": "Routine",
            "cleanliness_status": prediction,
            "confidence_score": f"{confidence:.2f}%"
        }

        output_pdf = "cleanliness_report.pdf"
        generate_pdf_report(report_data, image_path, output_pdf)

        print(f"Inspection Report saved as: {output_pdf}")

    except Exception as e:
        print(f"Error processing image: {e}")

def generate_pdf_report(data, image_path, output_pdf):
    """Creates a PDF inspection report with room image and classification results."""
    c = canvas.Canvas(output_pdf, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, height - 50, "Room Cleanliness Inspection Report")

    # General Information
    c.setFont("Helvetica", 12)
    c.drawString(100, height - 80, f"Inspection ID: {data['inspection_id']}")
    c.drawString(100, height - 100, f"Date of Inspection: {data['date']}")
    c.drawString(100, height - 120, f"Hotel Name: {data['hotel_name']}")
    c.drawString(100, height - 140, f"Room/Area Inspected: {data['room_area']}")
    c.drawString(100, height - 160, f"Inspection Type: {data['inspection_type']}")

    # Cleanliness Classification Summary
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, height - 200, "Cleanliness Classification Summary:")
    c.setFont("Helvetica", 12)
    c.drawString(100, height - 220, f"Overall Cleanliness Status: {data['cleanliness_status']}")
    c.drawString(100, height - 240, f"Confidence Score: {data['confidence_score']}")

    # Add Image
    if os.path.exists(image_path):
        img = ImageReader(image_path)
        c.drawImage(img, 100, height - 500, width=200, height=200)

    # Save PDF
    c.save()

# Example usage
predict_image('E:/Benchmark/Automated Inspection Report/images/test/0.png')'''