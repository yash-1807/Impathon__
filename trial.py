import numpy as np
import cv2

def homomorphic_edge_detection(image):
    # Step 1: Convert to grayscale and preprocess
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = np.float32(gray_image)

    # Step 2: Apply Log Transformation
    log_image = np.log1p(gray_image)

    # Step 3: Perform Fourier Transform
    f_transform = np.fft.fft2(log_image)

    # Step 4: Filter Design (e.g., Butterworth, Gaussian)
    # Example Butterworth High-pass filter
    rows, cols = gray_image.shape
    cy, cx = rows // 2, cols // 2
    radius = 30
    bw_filter = np.zeros((rows, cols), np.float32)
    for i in range(rows):
        for j in range(cols):
            dist = np.sqrt((i - cy)**2 + (j - cx)**2)
            bw_filter[i, j] = 1 - np.exp(-(dist**2) / (2 * (radius**2)))

    # Step 5: Apply Filtering
    filtered_image = f_transform * bw_filter

    # Step 6: Inverse Fourier Transform
    filtered_image = np.fft.ifft2(filtered_image)

    # Step 7: Exponential Transformation
    filtered_image = np.exp(np.real(filtered_image)) - 1

    # Step 8: Normalize
    enhanced_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX)

    # Step 9: Edge Detection
    edges = cv2.Canny(np.uint8(enhanced_image), 50, 150)

    return edges

# Example usage:
image = cv2.imread('input_image.jpg')
edges = homomorphic_edge_detection(image)
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()















---------------------------


Homomorphic filtering is a technique used in image processing to enhance contrast in images, particularly for images with varying illumination. It operates on the frequency domain of the image by manipulating the Fourier transform. Here's an algorithm to automatically detect and enhance edges in an image using homomorphic filtering:

Preprocessing:

Convert the input image from RGB to grayscale if it's not already in grayscale. Grayscale images are simpler to process and analyze for edge detection.
Ensure the image dimensions are a power of 2 for efficient Fourier transform operations.
Apply Log Transformation:

Apply a log transformation to the grayscale image to enhance the low-intensity details. This helps in reducing the influence of illumination variations in the image.
Perform Fourier Transform:

Compute the Fourier transform of the log-transformed grayscale image. This converts the image from the spatial domain to the frequency domain, where we can manipulate the frequencies corresponding to edges and illumination.
Filter Design:

Design a filter to separate the low-frequency (illumination) and high-frequency (edges) components. One common choice is a Butterworth or Gaussian filter.
Apply Filtering:

Apply the designed filter to the Fourier transformed image. This emphasizes the high-frequency components corresponding to edges while suppressing low-frequency components corresponding to illumination variations.
Inverse Fourier Transform:

Compute the inverse Fourier transform of the filtered image to return to the spatial domain.
Exponential Transformation:

Apply an exponential transformation to the result of the inverse Fourier transform. This reverses the log transformation applied earlier, restoring the original intensity scale.
Normalize:

Normalize the resulting image to ensure that pixel values are within the valid range (e.g., 0 to 255 for 8-bit grayscale images).
Edge Detection:

Apply an edge detection algorithm (e.g., Sobel, Canny, etc.) to the enhanced image to detect edges more accurately.
Post-processing:

Optionally, perform additional post-processing techniques such as noise reduction or morphological operations to refine the detected edges.



-------------------------------------------------------------
import cv2
import numpy as np

def preprocess(image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Ensure image dimensions are power of 2
    rows, cols = gray_image.shape
    new_rows = cv2.getOptimalDFTSize(rows)
    new_cols = cv2.getOptimalDFTSize(cols)
    padded_image = cv2.copyMakeBorder(gray_image, 0, new_rows - rows, 0, new_cols - cols, cv2.BORDER_CONSTANT, value=0)
    return padded_image

def homomorphic_filtering(image):
    # Apply log transformation
    image_log = np.log1p(np.float32(image))
    # Perform Fourier transform
    f_transform = np.fft.fft2(image_log)
    # Design Butterworth high-pass filter
    rows, cols = image.shape
    cy, cx = rows // 2, cols // 2
    radius = 30
    bw_filter = np.zeros((rows, cols), np.float32)
    for i in range(rows):
        for j in range(cols):
            dist = np.sqrt((i - cy) ** 2 + (j - cx) ** 2)
            bw_filter[i, j] = 1 - np.exp(-(dist ** 2) / (2 * (radius ** 2)))
    # Apply filtering
    filtered_image = f_transform * bw_filter
    # Inverse Fourier transform
    filtered_image = np.fft.ifft2(filtered_image)
    # Exponential transformation
    filtered_image = np.exp(np.real(filtered_image)) - 1
    return filtered_image

def edge_detection(image):
    # Convert to uint8
    image = np.uint8(image)
    # Apply Canny edge detection
    edges = cv2.Canny(image, 50, 150)
    return edges

def main():
    # Read input image
    image = cv2.imread('input_image.jpg')

    # Preprocess the image
    padded_image = preprocess(image)

    # Apply homomorphic filtering
    filtered_image = homomorphic_filtering(padded_image)

    # Crop the filtered image to original size
    cropped_filtered_image = filtered_image[:image.shape[0], :image.shape[1]]

    # Perform edge detection
    edges = edge_detection(cropped_filtered_image)

    # Display results
    cv2.imshow('Original Image', image)
    cv2.imshow('Enhanced Image', cropped_filtered_image)
    cv2.imshow('Edges', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
