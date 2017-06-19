from PIL import Image

from com.weijinglab.py.autolearning.common.MnistDataLoader import TrainingDataLoader
import numpy as np


# from common.MnistDataLoader import TrainingDataLoader
#
tdl = TrainingDataLoader("../common/auto_learning.properties")

img = tdl.get_train_data()[412]
img = img.reshape(28, 28)
print(np.uint8(img))
Image.fromarray(np.uint8(img)).show()
