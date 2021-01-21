import tensorflow as tf
import tensorflow_io as tfio
import matplotlib.pyplot as plt

# Program:  Color space conversion using TensorFlow
path = r'D:\Prag Python\lena.jpg'
img = tf.image.decode_jpeg(tf.io.read_file(path))

# Convert image to Gray Scale
gray_img = tfio.experimental.color.rgb_to_grayscale(img)

# Convert image to BGR color space
bgr_img = tfio.experimental.color.rgb_to_bgr(img)

# Convert image to CIEXYZ color space
img_temp = tf.cast(img, tf.float32) / 255.0 # require XYZ values in float32
xyz_img = tfio.experimental.color.rgb_to_xyz(img_temp)

# Convert image to YCbCr color space
ycbcr_img = tfio.experimental.color.rgb_to_ycbcr(img)

# Create subplots
fig = plt.figure()

ax1 = fig.add_subplot(2, 4, 1)
ax1.imshow(img, cmap='hot')
ax1.set_title('Color Image')
ax1.axis('off')

ax2 = fig.add_subplot(2, 4, 2)
ax2.imshow(gray_img, cmap='gray')
ax2.set_title('Gray Scale Image')
ax2.axis('off')

ax3 = fig.add_subplot(2, 4, 3)
ax3.imshow(bgr_img, cmap='hot')
ax3.set_title('BGR Image')
ax3.axis('off')

ax4 = fig.add_subplot(2, 4, 4)
ax4.imshow(xyz_img, cmap='hot' )
ax4.set_title('CIE XYZ Image')
ax4.axis('off')

ax5 = fig.add_subplot(2, 4, 5)
ax5.imshow(ycbcr_img, cmap='hot')
ax5.set_title('YCbCr Image')
ax5.axis('off')

ax6 = fig.add_subplot(2, 4, 6)
ax6.imshow(ycbcr_img[:,:,0], cmap='gray')
ax6.set_title('YCbCr Image - 1')
ax6.axis('off')

ax7 = fig.add_subplot(2, 4, 7)
ax7.imshow(ycbcr_img[:,:,1], cmap='gray')
ax7.set_title('YCbCr Image - 2')
ax7.axis('off')

ax8 = fig.add_subplot(2, 4, 8)
ax8.imshow(ycbcr_img[:,:,2], cmap='gray' )
ax8.set_title('YCbCr Image - 3')
ax8.axis('off')

plt.show()
