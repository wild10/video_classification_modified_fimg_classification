# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# avoid future tensorflow wargning
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

import argparse
import sys
import cv2 #opencv 2 for video
import numpy as np
import time
import tensorflow as tf


def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph


# The following two methods are taken from this tutorial:
# https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub

# Defining hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 64
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.io.decode_png(file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.io.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.io.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.io.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.compat.v1.Session()
  return sess.run(normalized)


def load_labels(label_file):
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  return [l.rstrip() for l in proto_as_ascii_lines]

# video classification


if __name__ == "__main__":

  doc_dir = "/home/predator/Documents/graymatics_test/"
  file_name =doc_dir + "tensorflow/tensorflow/examples/label_image/data/pizza.jpg"
  model_file = doc_dir +"tensorflow/tensorflow/examples/label_image/data/inception_v3_2016_08_28_frozen.pb"
  label_file = doc_dir +"tensorflow/tensorflow/examples/label_image/data/imagenet_slim_labels.txt"

  input_height = 299
  input_width = 299
  input_mean = 0
  input_std = 255
  input_layer = "input"
  output_layer = "InceptionV3/Predictions/Reshape_1"

  # hyperparameters for video
  video_path = doc_dir + "tensorflow/tensorflow/examples/label_image/data/neymar.mp4"
  writer = None
  screens_folder_path = "/home/predator/Documents/graymatics_test/tensorflow/tensorflow/examples/label_image/data/screens/"

  parser = argparse.ArgumentParser()
  parser.add_argument("--image", help="image to be processed")
  parser.add_argument("--graph", help="graph/model to be executed")
  parser.add_argument("--labels", help="name of file containing labels")
  parser.add_argument("--input_height", type=int, help="input height")
  parser.add_argument("--input_width", type=int, help="input width")
  parser.add_argument("--input_mean", type=int, help="input mean")
  parser.add_argument("--input_std", type=int, help="input std")
  parser.add_argument("--input_layer", help="name of input layer")
  parser.add_argument("--output_layer", help="name of output layer")
  args = parser.parse_args()

  if args.graph:
    model_file = args.graph
  if args.image:
    file_name = args.image
  if args.labels:
    label_file = args.labels
  if args.input_height:
    input_height = args.input_height
  if args.input_width:
    input_width = args.input_width
  if args.input_mean:
    input_mean = args.input_mean
  if args.input_std:
    input_std = args.input_std
  if args.input_layer:
    input_layer = args.input_layer
  if args.output_layer:
    output_layer = args.output_layer

  graph = load_graph(model_file)

  t = read_tensor_from_image_file(
      file_name,
      input_height=input_height,
      input_width=input_width,
      input_mean=input_mean,
      input_std=input_std)
   # print tensor of image
  # print('tensor img: {} shape: {}'.format(t,t.shape))

  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name)
  output_operation = graph.get_operation_by_name(output_name)

# CLASSIFICATION FOR A SINGLE IMAGE
  # with tf.compat.v1.Session(graph=graph) as sess:
  #
  #   results = sess.run(output_operation.outputs[0], {
  #       input_operation.outputs[0]: t
  #   })
  # results = np.squeeze(results)
  #
  # top_k = results.argsort()[-5:][::-1]
  # labels = load_labels(label_file)
  # print('size',len(labels) )
  # for i in top_k:
  #   print(labels[i], results[i])

  # CLASSIFICATION FOR VIDEOS 

  with tf.compat.v1.Session( graph = graph ) as sess:
      # video capture
      captured_video = cv2.VideoCapture(video_path)
      # number frames to capture
      number_frames = 120
      # captures some frames per second
      start = time.time() #start time
      for i in range(0, number_frames):
          ret, frame = captured_video.read()
      end = time.time() #end time
      seconds = end - start
      # print(' time taken : {} seconds'.format(seconds))
      fps = number_frames / seconds
      # print("Estimated frames per second : {0}".format(fps))

      i = 0
      while True:

          frame = captured_video.read()[1] #get currect frame
          frameId = captured_video.get(1)
          i = i + 1
          # print('frameId: {}', frameId)
          # write image into file to see samples
          cv2.imwrite(filename= screens_folder_path +str(i)+"_pizza.png", img=frame)
          # get the images saved
          image_data = tf.gfile.FastGFile(screens_folder_path+str(i)+"_pizza.png", 'rb' ).read()
          img_location = screens_folder_path+str(i)+"_pizza.png"

          # getting the tensor for the frame
          t = read_tensor_from_image_file(
            img_location,
            input_height=input_height,
            input_width=input_width,
            input_mean=input_mean,
            input_std=input_std
          )

          # get prediction for that image

          results = sess.run(output_operation.outputs[0], { input_operation.outputs[0]: t})

          # show prediction squeeze the dimmetion
          results = np.squeeze(results)
          top_k = results.argsort()[-2:][::-1]

          # top_k = results[0].argsort()[-len(results[0]):][::-1]

          labels = load_labels(label_file)
          pos = 1

          for k in top_k:
              # print(labels[k]  , results[k])
              label_text = labels[k]
              score = results[k]
              if( score >  0.8):
                  cv2.putText(frame, '%s  = %.5f' % (label_text, score),
                             (20, 20 * pos), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0))
              else:
                  cv2.putText(frame, '%s  = %.5f' % (label_text, score),
                         (20, 20 * pos), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255))
              pos = pos + 1

          if writer is None:
              # initialize our video writer
              fourcc = cv2.VideoWriter_fourcc(*"XVID")
              writer = cv2.VideoWriter("recognized.avi", fourcc, 10,
                  (frame.shape[1], frame.shape[0]), True)

          # write the output frame to disk
          writer.write(frame)

          cv2.imshow("image captured",frame)
          cv2.waitKey(1)
