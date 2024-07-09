import numpy as np
import cv2

def preprocess(image):
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   
    rows, cols = gray_image.shape
    new_rows = cv2.getOptimalDFTSize(rows)
    new_cols = cv2.getOptimalDFTSize(cols)
    padded_image = cv2.copyMakeBorder(gray_image, 0, new_rows - rows, 0, new_cols - cols, cv2.BORDER_CONSTANT, value=0)
    return padded_image

def homomorphic_filtering(image):
  
    image_log = np.log1p(np.float32(image))
    
    f_transform = np.fft.fft2(image_log)
    
    rows, cols = image.shape
    cy, cx = rows // 2, cols // 2
    radius = 30
    bw_filter = np.zeros((rows, cols), np.float32)
    for i in range(rows):
        for j in range(cols):
            dist = np.sqrt((i - cy)**2 + (j - cx)**2)
            bw_filter[i, j] = 1 - np.exp(-(dist**2) / (2 * (radius**2)))
   
    filtered_image = f_transform * bw_filter
   
    filtered_image = np.fft.ifft2(filtered_image)
    
    filtered_image = np.exp(np.real(filtered_image)) - 1
    return filtered_image

def edge_detection(image):
   
    image = np.uint8(image)
    
    edges = cv2.Canny(image, 50, 150)
    return edges

def main():
    
    image = cv2.imread('1.jpg')

 
    padded_image = preprocess(image)

    
    filtered_image = homomorphic_filtering(padded_image)

    
    cropped_filtered_image = filtered_image[:image.shape[0], :image.shape[1]]

   
    edges = edge_detection(cropped_filtered_image)

    
    cv2.imshow('Original Image', image)
    cv2.imshow('Enhanced Image', cropped_filtered_image)
    cv2.imshow('Edges', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
