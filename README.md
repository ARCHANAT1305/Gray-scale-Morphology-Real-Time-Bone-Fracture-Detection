# Gray-scale-Morphology-Real-Time-Bone-Fracture-Detection
## AIM: 
To detect the gray scale morphology rel time bone fracture detection.
## ALGORITHM:
### STEP 1:
Upload the image
### STEP 2:
Read the uploaded image using cv2.imread().
### STEP 3:
Convert the image to grayscale to simplify processing.Apply Gaussian blur to reduce noise and smooth the image.
### STEP 4:
Perform morphological closing to fill small gaps and enhance edges.Detect edges using the Canny edge detection algorithm.
### STEP 5:
Find contours from the detected edges.Draw the contours on a copy of the original image.
### STEP 6:
Convert BGR to RGB for correct color display in Matplotlib.
### STEP 7:
Display original, edge-detected, and contoured images side by side using Matplotlib.


## PROGRAM:
### NAME: ARCHANA T
### REGISTER NUMBER :212223240013

```
from google.colab import files
import cv2
import numpy as np
import matplotlib.pyplot as plt
uploaded = files.upload()
image_path = next(iter(uploaded))

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred
def detect_bone_edges(preprocessed, original):
   
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(preprocessed, cv2.MORPH_CLOSE, kernel)
    edges = cv2.Canny(morph, 30, 100)

    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = original.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    return result, edges
def present_results(original_image, processed_image, edges):
    original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    processed_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original_rgb)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Detected Edges")
    plt.imshow(edges, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Contour Detection")
    plt.imshow(processed_rgb)
    plt.axis('off')

    plt.show()
image = cv2.imread(image_path)

if image is None:
    print("❌ Error: Image not found. Check the uploaded file name.")
else:
    preprocessed = preprocess_image(image)
    bone_detected_image, edges = detect_bone_edges(preprocessed, image)
    present_results(image, bone_detected_image, edges)
```


## OUTPUT:
<img width="1181" height="520" alt="image" src="https://github.com/user-attachments/assets/7c30bf89-3194-41f0-8190-c8052967f453" />

## RESULT:
Thus the program to detect the gray scale morphology rel time bone fracture detection using python was executed successfully.
