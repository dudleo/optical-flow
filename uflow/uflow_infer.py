

import gin
from absl import app
from absl import flags
from uflow import uflow_flags

from uflow.uflow_net import UFlow
from uflow.uflow_plotting import flow_to_rgb

import tensorflow as tf
import numpy as np

import cv2

FLAGS = flags.FLAGS

def create_uflow():
  occ_weights = {
      'fb_abs': FLAGS.occ_weights_fb_abs,
      'forward_collision': FLAGS.occ_weights_forward_collision,
      'backward_zero': FLAGS.occ_weights_backward_zero,
  }
  # Switch off loss-terms that have weights < 1e-2.
  occ_weights = {k: v for (k, v) in occ_weights.items() if v > 1e-2}

  occ_thresholds = {
      'fb_abs': FLAGS.occ_thresholds_fb_abs,
      'forward_collision': FLAGS.occ_thresholds_forward_collision,
      'backward_zero': FLAGS.occ_thresholds_backward_zero,
  }
  occ_clip_max = {
      'fb_abs': FLAGS.occ_clip_max_fb_abs,
      'forward_collision': FLAGS.occ_clip_max_forward_collision,
  }

  uflow = UFlow(
      checkpoint_dir=FLAGS.checkpoint_dir,
      optimizer=FLAGS.optimizer,
      #learning_rate=learning_rate_fn,
      only_forward=FLAGS.only_forward,
      level1_num_layers=FLAGS.level1_num_layers,
      level1_num_filters=FLAGS.level1_num_filters,
      level1_num_1x1=FLAGS.level1_num_1x1,
      dropout_rate=FLAGS.dropout_rate,
      #build_selfsup_transformations=build_selfsup_transformations,
      fb_sigma_teacher=FLAGS.fb_sigma_teacher,
      fb_sigma_student=FLAGS.fb_sigma_student,
      train_with_supervision=FLAGS.use_supervision,
      train_with_gt_occlusions=FLAGS.use_gt_occlusions,
      smoothness_edge_weighting=FLAGS.smoothness_edge_weighting,
      teacher_image_version=FLAGS.teacher_image_version,
      stop_gradient_mask=FLAGS.stop_gradient_mask,
      selfsup_mask=FLAGS.selfsup_mask,
      normalize_before_cost_volume=FLAGS.normalize_before_cost_volume,
      original_layer_sizes=FLAGS.original_layer_sizes,
      shared_flow_decoder=FLAGS.shared_flow_decoder,
      channel_multiplier=FLAGS.channel_multiplier,
      num_levels=FLAGS.num_levels,
      use_cost_volume=FLAGS.use_cost_volume,
      use_feature_warp=FLAGS.use_feature_warp,
      accumulate_flow=FLAGS.accumulate_flow,
      occlusion_estimation=FLAGS.occlusion_estimation,
      occ_weights=occ_weights,
      occ_thresholds=occ_thresholds,
      occ_clip_max=occ_clip_max,
      smoothness_at_level=FLAGS.smoothness_at_level,
  )
  return uflow

import threading
from threading import Lock
import cv2

class Camera:
    last_frame = None
    last_ready = None
    lock = Lock()

    def __init__(self, rtsp_link):
        capture = cv2.VideoCapture(rtsp_link)
        thread = threading.Thread(target=self.rtsp_cam_buffer, args=(capture,), name="rtsp_read_thread")
        thread.daemon = True
        thread.start()

    def rtsp_cam_buffer(self, capture):
        while True:
            with self.lock:
                self.last_ready, self.last_frame = capture.read()


    def getFrame(self):
        if (self.last_ready is not None) and (self.last_frame is not None):
            return self.last_frame.copy()
        else:
            return None

class CameraUSB:
    def __init__(self):
        self.capture = cv2.VideoCapture(0)

    def getFrame(self):
        _, frame = self.capture.read()

        return frame


def main(unused_args):
    # _pcm.sdp or _ulaw.sdp

    #gin.parse_config_files_and_bindings(FLAGS.config_file, FLAGS.gin_bindings)

    uflow = create_uflow()
    #FLAGS.checkpoint_dir = '../models'
    FLAGS.init_checkpoint_dir = '../models'
    uflow.update_checkpoint_dir(FLAGS.init_checkpoint_dir)
    uflow.restore(
        reset_optimizer=FLAGS.reset_optimizer,
        reset_global_step=FLAGS.reset_global_step)
    #uflow.update_checkpoint_dir(FLAGS.checkpoint_dir)

    #1242 x 375

    #cap = cv2.VideoCapture("rtsp://192.168.0.100:8080/h264_ulaw.sdp")
    #cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # _ulaw or _pcm
    #cap = Camera("rtsp://192.168.0.100:8080/h264_pcm.sdp")
    cap = CameraUSB()
    # check camera
    ''''
    while True:
        img = None
        while img is None:
            img = cap.getFrame()
        cv2.imshow('frame', img)
        cv2.waitKey(1)
    '''
    while True:
        img1 = None
        img2 = None
        while img1 is None:
            img1 = cap.getFrame()
        img1 = img1.astype('float32') / 255.
        cv2.waitKey(33)
        while img2 is None:
            img2 = cap.getFrame()
        img2 = img2.astype('float32') / 255.
        # img = cv2.resize(img, (1242, 375))
        #cv2.imshow('frame', img)
        #cv2.waitKey(1)

        #img1 = cv2.imread('files/00000.jpg').astype('float32') / 255.
        #img2 = cv2.imread('files/00001.jpg').astype('float32') / 255.

        #img1 = img1.resize(height=640, width=640 )
        #img2 = img2.resize(height=640, width=640 )
        optical_flow = uflow.infer(img1, img2)
        optical_flow = optical_flow.numpy()
        optical_flow = flow_to_rgb(optical_flow)
        cv2.imshow('flow', np.vstack((img1, optical_flow)))
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    app.run(main)