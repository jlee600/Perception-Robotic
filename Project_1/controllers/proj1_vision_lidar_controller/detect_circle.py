import cv2
import numpy as np

def detect_circle(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a color threshold to create a mask for white regions
    # Adjust the lower and upper bounds as needed
    lower_white = np.array([200, 200, 200], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(image, lower_white, upper_white)

    # cv2.imshow("Mask", mask)

    # Apply the mask to the grayscale image
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)

    # Apply median blur to the masked grayscale image
    masked_gray = cv2.medianBlur(masked_gray, 5)

    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(masked_gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                            param1=100, param2=25, minRadius=30, maxRadius=200)

    # If some circles are detected, print their coordinates and draw them
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(image, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    
    return circles

if __name__ == "__main__":
    image = cv2.imread("test.jpg")
    circle = detect_circle(image)
    print(circle)