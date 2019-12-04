from django.db import models
from django.urls import reverse

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import json
import time
import glob
from io import StringIO
from PIL import Image
import matplotlib.pyplot as plt
from detection.utils import visualization_utils as vis_util
from detection.utils import label_map_util
from multiprocessing.dummy import Pool as ThreadPool
import sys

from django.conf import settings


class Detection(models.Model):
    title = models.CharField(max_length=250)
    img_original = models.ImageField(upload_to='galaxies_img')
    img = models.ImageField(upload_to='galaxies_detection', blank=True)
    spiral_count = models.IntegerField(null=True)
    elliptical_count = models.IntegerField(null=True)
    irregular_count = models.IntegerField(null=True)

    def get_absolute_url(self):
        return reverse('detail', kwargs={'pk': self.pk})



    def save(self, *args, **kwargs):
        super(Detection, self).save(*args, **kwargs)

        filename = os.path.abspath(os.path.join(settings.MEDIA_ROOT, self.img_original.path))
        
        MAX_NUMBER_OF_BOXES = 100
        MINIMUM_CONFIDENCE = 0.4

        PATH_TO_LABELS = 'detection/config/label_map.pbtxt'

        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=sys.maxsize, use_display_name=True)
        CATEGORY_INDEX = label_map_util.create_category_index(categories)

        PATH_TO_CKPT = 'detection/model/frozen_inference_graph.pb'

        def load_image_into_numpy_array(image):
            (im_width, im_height) = image.size
            return np.array(image.getdata()).reshape(
                (im_height, im_width, 3)).astype(np.uint8)

        def detect_objects(image_path):

            spiral = 0
            elliptical = 0
            irregular = 0

            image = Image.open(image_path)
            image_np = load_image_into_numpy_array(image)
            image_np_expanded = np.expand_dims(image_np, axis=0)

            (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: image_np_expanded})

            _classes = np.squeeze(classes).astype(np.int32)

            _scores = np.squeeze(scores)

            scores_res = list(_scores)



            for x in scores_res:
              
                if x >= 0.4:
                    print(f'index: {scores_res.index(x)}, count {x}')                    

                    cl = _classes[scores_res.index(x)]

                    if cl == 1:
                        spiral += 1
                    if cl == 2:
                        elliptical += 1
                    if cl == 3:
                        irregular += 1



            print(f'spiral: {spiral}')
            print(f'elliptical: {elliptical}')
            print(f'irregular: {irregular}')

            self.spiral_count = spiral
            self.elliptical_count = elliptical
            self.irregular_count = irregular
            

            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                CATEGORY_INDEX,
                min_score_thresh=MINIMUM_CONFIDENCE,
                use_normalized_coordinates=True,
                line_thickness=5)
            fig = plt.figure()
            fig.set_size_inches(16, 9)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)


            filename2 = os.path.abspath(os.path.join(settings.MEDIA_ROOT+'/galaxies_detection/'))


            plt.imshow(image_np, aspect = 'auto')
            plt.savefig('{}/{}'.format(filename2, 'label'+str(self.pk)+'.jpg'), dpi = 300)
            print(boxes)

            path_dj = '/galaxies_detection/'
            self.img = path_dj + 'label'+str(self.pk)+'.jpg'

            plt.close(fig)

        



        print('Loading model...')
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        print('detecting...')
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        
                detect_objects(filename)

        super().save()





