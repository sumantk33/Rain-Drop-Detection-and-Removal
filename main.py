import cv2
import numpy as np

image_name = 'rain.png'

img_raw = cv2.imread(image_name, 1)

# Converting the image to grayscale
img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)

# K-Means clustering methods
def K_Means(Image, K):

    # Reshaping the image based on the color scale
    if(len(Image.shape) < 3):
        Z = Image.reshape((-1, 1))
    elif len(Image.shape) == 3:
        Z = Image.reshape((-1, 3))
    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(
        Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    Clustered_Image = res.reshape((Image.shape))

    return Clustered_Image


Clusters = 3
Clustered_Image = K_Means(img, Clusters)

# Creating the Canny image
canny = cv2.Canny(Clustered_Image, 50, 150)
drop_image = canny.copy()

# Hough transform to detect circles of rain droplets
circles = cv2.HoughCircles(drop_image, cv2.HOUGH_GRADIENT, 1, 1,
                           param1=150, param2=45, minRadius=0, maxRadius=0)

# Converting x,y and r into integer type
# (x,y) -> co-ordinates of circle centre
# r -> radius of the circle
detected_circles = np.uint16(np.around(circles))

# Drawing circle in the image on the rain droplets
for (x, y, r) in detected_circles[0, :]:
    cv2.circle(drop_image, (x, y), r, (0, 255, 0), 1)

# Converting the drop_image into RGB
backtorgb = cv2.cvtColor(drop_image, cv2.COLOR_GRAY2RGB)
# Superimposing droplets image on the original image
final_image = img_raw + backtorgb

# Applying median filter
result = cv2.medianBlur(final_image, 7)

cv2.imshow('Input_image', img_raw)
cv2.imshow('Clustered_Image', Clustered_Image)
cv2.imshow('Canny_Image', canny)
cv2.imshow('drop_image', drop_image)
cv2.imshow('result', result)


cv2.waitKey(0)
cv2.destroyAllWindows()
