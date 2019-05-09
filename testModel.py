import cv2
import numpy as np
import tensorflow as tf

import alexnet
import caffe_classes


class AlexNetModel(object):

    def recognize(self, img_path):
        img = cv2.imread(img_path)
        dropoutPro = 1
        classNum = 1000
        skip = []

        imgMean = np.array([104, 117, 124], np.float)
        x = tf.placeholder("float", [1, 227, 227, 3])
        model = alexnet.alexNet(x, dropoutPro, classNum, skip)
        score = model.fc3
        softmax = tf.nn.softmax(score)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            model.loadModel(sess)  # Load the model

            test = cv2.resize(img.astype(float), (227, 227))  # resize
            test -= imgMean  # subtract image mean
            test = test.reshape((1, 227, 227, 3))  # reshape into tensor shape
            maxx = np.argmax(sess.run(softmax, feed_dict={x: test}))
            res = caffe_classes.class_names[maxx]  # find the max probility
            print(res)
            return res
