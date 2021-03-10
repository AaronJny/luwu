# -*- coding: utf-8 -*-
# @Author       : AaronJny
# @LastEditTime : 2021-03-07
# @FilePath     : /LuWu/luwu/core/models/complex/od/templates/project/eval.py
# @Desc         :
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from PIL import Image

warnings.filterwarnings("ignore")  # Suppress Matplotlib warnings

PATH_TO_SAVED_MODEL = "my_model/saved_model"
PATH_TO_LABEL_MAP = "label_map.pbtxt"
PATH_TO_EVAL_IMAGES = [
    "test.image",
]


# 加载模型到内存
print("Loading model...", end="")
start_time = time.time()
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
end_time = time.time()
elapsed_time = end_time - start_time
print("Done! Took {} seconds".format(elapsed_time))

# 加载 label_map.pbtxt
category_index = label_map_util.create_category_index_from_labelmap(
    PATH_TO_LABEL_MAP, use_display_name=True
)

# 调用模型对指定图片进行目标检测
def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))


for image_path in PATH_TO_EVAL_IMAGES:
    print("Running inference for {}... ".format(image_path), end="")
    image_np = load_image_into_numpy_array(image_path)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]
    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)
    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop("num_detections"))
    detections = {
        key: value[0, :num_detections].numpy() for key, value in detections.items()
    }
    # detections是一个dict,结构如下：
    # - num_detections: a tf.int tensor with only one value, the number of detections [N].
    # - detection_boxes: a tf.float32 tensor of shape [N, 4] containing bounding box coordinates in the following order: [ymin, xmin, ymax, xmax].
    # - detection_classes: a tf.int tensor of shape [N] containing detection class index from the label file.
    # - detection_scores: a tf.float32 tensor of shape [N] containing detection scores.
    # - raw_detection_boxes: a tf.float32 tensor of shape [1, M, 4] containing decoded detection boxes without Non-Max suppression. M is the number of raw detections.
    # - raw_detection_scores: a tf.float32 tensor of shape [1, M, 90] and contains class score logits for raw detection boxes. M is the number of raw detections.
    # - detection_anchor_indices: a tf.float32 tensor of shape [N] and contains the anchor indices of the detections after NMS.
    # - detection_multiclass_scores: a tf.float32 tensor of shape [1, N, 90] and contains class score distribution (including background) for detection boxes in the image including background class.
    detections["num_detections"] = num_detections

    # detection_classes should be ints.
    detections["detection_classes"] = detections["detection_classes"].astype(np.int64)

    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections["detection_boxes"],
        detections["detection_classes"],
        detections["detection_scores"],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=0.30,
        agnostic_mode=False,
    )

    plt.figure()
    plt.imshow(image_np_with_detections)
    print("Done")
plt.show()
