# face Mask Detection and Masked Face Recognition 


The goal of this project is to develop a system for face mask detection and masked face recognition. The system utilizes computer vision techniques and deep learning models to detect whether a person is wearing a face mask or not, and also perform face recognition on masked faces. The project aims to contribute to the ongoing efforts in promoting public health and safety by enforcing face mask usage and identifying individuals even when their faces are partially covered.

## Methodology

The project utilizes the following methodology:
### 1. Face Mask Detection

The face mask detection component is responsible for identifying whether a person in the captured image or video is wearing a face mask or not. The methodology employed for face mask detection consists of the following steps:

    Preprocessing: The input frame is resized and converted to the appropriate format for further processing.

    Face Detection: A pre-trained face detection model based on the Single Shot MultiBox Detector (SSD) architecture is used to detect faces in the frame.

    Region of Interest (ROI) Extraction: The bounding box coordinates of the detected face are used to extract the region of interest (ROI) containing the face.

    Face Mask Classification: A pre-trained deep learning model is employed to classify the ROI as either "with mask" or "without mask".

    Output Visualization: The results of the face mask classification are visualized on the frame by drawing bounding boxes around the detected faces and displaying the corresponding label.

### 2. Masked Face Recognition

The masked face recognition component aims to identify individuals even when they are wearing face masks. The methodology for masked face recognition involves the following steps:

    Dataset Preparation: The dataset includes images of different individuals in different subfolders. 

    Face Encoding: The face recognition library is utilized to extract facial encodings from the images in the dataset. These encodings capture unique facial features.The pretrained face detection models detecting faces in the input video frames. It processes the frames using deep learning techniques, analyzes the image data, and identifies the regions that potentially contain faces. These detected face regions are then passed to the face mask detection model for further analysis and prediction.

    Real-Time Face Recognition: The face encodings obtained from the dataset are compared with the encodings of faces detected in real-time. The library performs face matching and returns the most similar face based on the encodings.

    Labeling and Visualization: The recognized face is labeled with the corresponding individual's name, and the result is displayed on the frame.

## Libraries

    Python 3.x
    OpenCV
    face_recognition
    Keras with TensorFlow backend
    imutils
    numpy
    pygame
    