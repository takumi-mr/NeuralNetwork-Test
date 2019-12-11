import mnist
import numpy as np
from PIL import Image

def showImg(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

train_images = mnist.train_images()
train_labels = mnist.train_labels()

test_images = mnist.test_images()
test_labels = mnist.test_labels()

img = train_images[1]
label = train_labels[1]
print(label)
print(img.shape)
showImg(img)
