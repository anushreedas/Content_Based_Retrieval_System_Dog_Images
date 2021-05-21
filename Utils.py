import numpy as np
import tensorflow as tf

def get_random():
    test_ids = []
    with open('test_files.csv', 'r') as csvFile:
        data = csvFile.readlines()
    for row in data:
        row = row.strip()
        test_ids.append(row)

    index = np.random.choice(len(test_ids))
    image_path = test_ids[index]
    return image_path


def segment_dog(image_og,seg_model):
    image_t = tf.image.resize(image_og, (128, 128))
    image_t = tf.cast(image_t, tf.float32) / 255.
    image_t = image_t[tf.newaxis, ...]
    pred_mask = seg_model.predict(image_t)
    mask = tf.image.resize(pred_mask[0], (image_og.shape[0], image_og.shape[1]))
    mask = np.array(mask[:, :, 1])
    mask = mask.clip(max=1)
    mask = mask.clip(min=0)
    mask = np.array(mask * 255).astype('uint8')
    mask = 255 - mask
    return mask.astype(np.uint8)
