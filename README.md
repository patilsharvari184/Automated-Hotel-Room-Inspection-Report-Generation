# Automated-Inspection-Report-Generation-Hotel-Rooms-

# Room Cleanliness Inspection System

This repository contains a deep learning model for classifying room images as "Clean" or "Messy" and a Flask web application to demonstrate its functionality. The system can generate PDF inspection reports based on the predictions.

## Table of Contents

  * [Features]
  * [Project Structure]
  * [Setup and Installation]
      * [Prerequisites]
      * [Installation]
  * [Usage]
      * [Training the Model]
      * [Running the Web Application]
  * [Model Details]
  * [Report Generation]
  * [Contributing]
  * [License]

## Features

  * **Image Classification:** Classifies room images as "Clean" or "Messy" using a VGG16-based convolutional neural network.
  * **Web Interface:** A user-friendly Flask application for uploading images and viewing predictions.
  * **PDF Report Generation:** Generates comprehensive PDF inspection reports with details like inspection ID, date, property information, cleanliness status, and confidence score.

## Project Structure

```
.
├── main.py                 # Script for training the deep learning model and initial report generation
├── app.py                  # Flask web application for image prediction and report generation
├── requirements.txt        # Python dependencies
├── model1.h5               # Pre-trained model (will be generated after running main.py or downloaded)
└── static/
    ├── uploads/            # Directory for uploaded images via the web app
    ├── reports/            # Directory for generated PDF reports
    └── css/                # CSS files for the web app (if any)
    └── js/                 # JavaScript files for the web app (if any)
└── templates/
    ├── index.html          # HTML template for the main upload page
    └── result.html         # HTML template for displaying prediction results and report download link
└── hotel_dataset/          # Dataset directory (assumed structure)
    ├── Dataset/
        ├── train/
            ├── clean/
            └── messy/
        ├── val/
            ├── clean/
            └── messy/
```

## Setup and Installation

### Prerequisites

  * Python 3.8+
  * pip (Python package installer)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Dataset:**
    Place your image dataset in the `hotel_dataset/Dataset/` directory with `train` and `val` subdirectories, each containing `clean` and `messy` subfolders for your images, as indicated in `main.py`. For example:

    ```
    hotel_dataset/
    └── Dataset/
        ├── train/
        │   ├── clean/
        │   └── messy/
        └── val/
            ├── clean/
            └── messy/
    ```

## Usage

### Training the Model

The `main.py` script handles the training of the VGG16-based model. It will save the trained model as `my_model22.h5`.

1.  **Ensure your dataset is correctly placed** as described in the [Dataset](https://www.google.com/search?q=%23dataset) section.

2.  **Run the training script:**

    ```bash
    python main.py
    ```

    This will start the training process, display accuracy and loss plots, and save the model as `my_model22.h5`. You might want to rename this to `model1.h5` or update `app.py` to load `my_model22.h5`.

### Running the Web Application

The `app.py` script runs the Flask web application.

1.  **Ensure you have a trained model file** (`model1.h5` or `my_model22.h5` renamed to `model1.h5`) in the root directory.

2.  **Run the Flask application:**

    ```bash
    python app.py
    ```

3.  Open your web browser and navigate to `http://127.0.0.1:5000` (or `http://localhost:5000`).

4.  Upload an image to get a cleanliness prediction and generate a PDF report.

## Model Details

The model used is a fine-tuned VGG16 convolutional neural network.

  * **Base Model:** VGG16 pre-trained on ImageNet, with its layers frozen initially.
  * **Custom Layers:**
      * `GlobalAveragePooling2D()`: Reduces dimensionality.
      * `Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))`: A dense layer with L2 regularization.
      * `Dropout(0.5)`: Dropout layer for regularization.
      * `Dense(2, activation='softmax')`: Output layer for two classes (Clean/Messy).
  * **Optimizer:** Adam with a learning rate of `0.00001`.
  * **Loss Function:** `binary_crossentropy`.
  * **Callbacks:**
      * `EarlyStopping`: Monitors validation loss and stops training if it doesn't improve for 7 consecutive epochs.
      * `ModelCheckpoint`: (Commented out in `main.py`'s final `model.fit` call but present in definition) Saves model checkpoints.

## Report Generation

The system generates a detailed PDF report for each inspection, including:

  * Inspection ID
  * Date of Inspection
  * Property Name
  * Property Region
  * Property Type
  * Service
  * Room/Area Inspected
  * Inspection Type
  * Overall Cleanliness Status (Clean/Messy)
  * Confidence Score
  * The inspected image

Reports are saved in the `static/reports/` directory and can be downloaded directly from the web application's result page.

## Contributing

Contributions are welcome\! Please feel free to:

  * Fork the repository.
  * Create new branches for features or bug fixes.
  * Submit pull requests.

## License

This project is open-source and available under the MIT License.
