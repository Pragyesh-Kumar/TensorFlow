import tensorflow as tf
import tensorflow as tfio
import matplotlib.pyplot as plt
import cv2

# Image read and Display using TensorFlow
path = r'D:\Prag Python\lena.jpg'
img = tf.image.decode_jpeg(tf.io.read_file(path))

plt.figure()
plt.title("Image Display using TF")
plt.imshow(img)
plt.axis('off')
plt.show()

# Image read and Display using OpenCV
img_cv = cv2.imread(path)
cv2.imshow("Image Display using OpenCV", img_cv)
cv2.waitKey(0)
