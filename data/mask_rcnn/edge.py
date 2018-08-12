import os
import sys
import numpy as np
from scipy.signal import convolve2d
from scipy.misc import imsave

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# Root directory of the project
ROOT_DIR = os.path.abspath("mask_rcnn/")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# # Download COCO trained weights from Releases if needed
# if not os.path.exists(COCO_MODEL_PATH):
#     utils.download_trained_weights(COCO_MODEL_PATH)

# # Directory of images to run detection on
# IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

def getEdge(image):
	results = model.detect([image], verbose=0)
	r = results[0]
	H,W,_ = r['masks'].shape
	if H == 28:
		return np.zeros((128, 416, 3))
	im = np.zeros((H, W, 3))

	for i in range(r['masks'].shape[2]):
	    mask = r['masks'][:,:,i]
	    him = convolve2d(mask, [[1, 2, 1],[0, 0, 0],[-1, -2, -1]], mode="same")
	    vim = convolve2d(mask, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], mode="same")
	    grad = (him*him+vim*vim)**0.5
	    im[:,:,0] += grad
	    im[:,:,1] += grad
	    im[:,:,2] += grad
	if np.max(im) != 0:
		im /= np.max(im)
	return im * 256