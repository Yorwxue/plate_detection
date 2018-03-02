import argparse
import os
import time
import subprocess

import cv2
import numpy as np
import json
import model
import tensorflow as tf
from tqdm import tqdm

from yolov2.preprocessing import parse_annotation
from yolov2.utils import draw_boxes
from yolov2.frontend import YOLO

from eval import resize_image, detect, sort_poly


config_path = "yolov2/config.json"
weights_path = "yolov2/best_weights_motorbike.h5"

east_checkpoint_path = "checkpoint/east_icdar2015_resnet_v1_50_rbox"

tessdata_path = '/media/clliao/006a3168-df49-4b0a-a874-891877a88870/clliao/workspace/open-source/tesseract-ocr/tessdata'


# tesseract
def image_to_string(img, cleanup=True, plus=''):  # ex plus='--psm 7'
    image_name = 'tmp.png'
    txt_name = 'tmp'
    cv2. imwrite(image_name, img)
    os.system('export TESSDATA_PREFIX=%s \n' % tessdata_path + 'tesseract %s %s %s' % (plus, image_name, txt_name))
    # os.system('tesseract %s %s %s' % (plus, image_name, txt_name))
    # subprocess.run('export TESSDATA_PREFIX=%s' % tessdata_path)
    # subprocess.check_output('tesseract %s %s %s' % (plus, image_name, txt_name), shell=True)
    text = ''
    with open(txt_name + '.txt', 'r') as f:
        text = f.read().strip()
    if cleanup:
        os.remove(txt_name + '.txt')
        os.remove(image_name)
    return text


# image_path   = "../test/1.png"
with open(config_path) as config_buffer:
    config = json.load(config_buffer)

###############################
#   Make the model
###############################

# yolo = YOLO(architecture=config['model']['architecture'],
#             input_size=config['model']['input_size'],
#             labels=config['model']['labels'],
#             max_box_per_image=config['model']['max_box_per_image'],
#             anchors=config['model']['anchors'])

###############################
#   Load trained weights
###############################
# yolo.load_weights(weights_path)

# show variables
# for v in tf.global_variables():
#     print(v)
# exit()

video_path = '/media/clliao/9c88dfb2-c12d-48cc-b30b-eaffb0cbf545/street_videos/new_camera/vlc-record-2018-03-01-10h25m02s-rtsp___10.10.53.211_554_live.sdp-.mp4'
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (1280,720))

# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#
#     if not ret:
#         break
#
#     # Our operations on the frame come here
#     # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # yolo v2
#     yolo_boxes = yolo.predict(frame)
#     image = draw_boxes(frame, yolo_boxes, config['model']['labels'])
#     # Display the resulting frame
#     # cv2.imshow('frame', gray)
#     cv2.imshow('frame', image)
#     cv2.waitKey(1)
#
#     # if cv2.waitKey(1) & 0xFF == ord('q'):
#     #     break
#
#     # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()
#
# exit()
# ------------------------------------------------------------------------------------------------------------------------
with tf.get_default_graph().as_default():
    input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    # east
    f_score, f_geometry = model.model(input_images, is_training=False)
    variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
    restorer = tf.train.Saver(variable_averages.variables_to_restore())

    #  yolo
    yolo = YOLO(architecture=config['model']['architecture'],
                input_size=config['model']['input_size'],
                labels=config['model']['labels'],
                max_box_per_image=config['model']['max_box_per_image'],
                anchors=config['model']['anchors'])

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        load_model_time = time.time()
        yolo.load_weights(weights_path)
        ckpt_state = tf.train.get_checkpoint_state(east_checkpoint_path)
        model_path = os.path.join(east_checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
        restorer.restore(sess, model_path)
        print('Load model spent %f sec' % (time.time() - load_model_time))

        frame_no = 0
        try:
            while(True):
                # Capture frame-by-frame
                frame_no += 1
                ret, frame = cap.read()

                if not ret:
                    break

                # Our operations on the frame come here
                # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # yolo v2
                yolo_boxes = yolo.predict(frame)
                image = frame
                image = draw_boxes(image, yolo_boxes, config['model']['labels'])

                # east for car license plate detection
                for yolo_box in yolo_boxes:
                    yolo_xmin = np.max([int((yolo_box.x - yolo_box.w / 2) * frame.shape[1]), 0])
                    yolo_xmax = np.min([int((yolo_box.x + yolo_box.w / 2) * frame.shape[1]), frame.shape[1]])
                    yolo_ymin = np.max([int((yolo_box.y - yolo_box.h / 2) * frame.shape[0]), 0])
                    yolo_ymax = np.min([int((yolo_box.y + yolo_box.h / 2) * frame.shape[0]), frame.shape[0]])
                    crop_img = frame[yolo_ymin:yolo_ymax, yolo_xmin:yolo_xmax, :]

                    # cv2.imshow('frame', frame)
                    # cv2.imshow('crop', crop_img)
                    # cv2.waitKey(0)
                    # exit()

                    im_resized, (ratio_h, ratio_w) = resize_image(crop_img)

                    timer = {'net': 0, 'restore': 0, 'nms': 0}
                    start = time.time()
                    score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: [im_resized]})
                    timer['net'] = time.time() - start

                    east_boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)

                    # frame[0:100, 0:200, :] = np.zeros((100, 200, 3))

                    if east_boxes is not None:
                        east_boxes = east_boxes[:, :8].reshape((-1, 4, 2))
                        east_boxes[:, :, 0] /= ratio_w
                        east_boxes[:, :, 1] /= ratio_h

                        for east_box in east_boxes:
                            # to avoid submitting errors
                            east_box = sort_poly(east_box.astype(np.int32))
                            if np.linalg.norm(east_box[0] - east_box[1]) < 5 or np.linalg.norm(east_box[3] - east_box[0]) < 5:
                                continue

                            # box boundary
                            east_xmin = np.min(east_box[:, 0])
                            east_xmax = np.max(east_box[:, 0])
                            east_ymin = np.min(east_box[:, 1])
                            east_ymax = np.max(east_box[:, 1])

                            # tesseract Optical Character Recognition
                            #plate_string = 'Not Found'
                            # plate_string = image_to_string(crop_img[np.min([0, east_ymin-10]): east_ymax+10,
                            #                                np.min([0, east_xmin-10]): east_xmax+10])

                            # print(plate_number)

                            # cv2.imwrite('./plate/img_%s.png' % str(frame_no), crop_img[east_ymin:east_ymax, east_xmin:east_xmax])

                            # show plate
                            frame[0:100, 0:200, :] = cv2.resize(crop_img[east_ymin:east_ymax, east_xmin:east_xmax], (200, 100))

                            # mark plate
                            # -----------
                            LicensePlate = east_box.astype(np.int32).reshape((-1, 1, 2))
                            LicensePlate[1] = LicensePlate[1] - LicensePlate[0]
                            LicensePlate[2] = LicensePlate[2] - LicensePlate[0]
                            LicensePlate[3] = LicensePlate[3] - LicensePlate[0]
                            LicensePlate[0] = [0,0]

                            # ----------
                            cv2.polylines(crop_img, [east_box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0),
                                          thickness=1)
                            cv2.imwrite("./plate/"+time.strftime('%Y-%m-%d %H:%M:%S')+".jpg",crop_img[east_ymin:east_ymax, east_xmin:east_xmax])
                            frame[yolo_ymin:yolo_ymax, yolo_xmin:yolo_xmax, :] = crop_img

                            font = cv2.FONT_HERSHEY_SIMPLEX
                            #cv2.putText(frame, str(plate_string), (int(frame.shape[1]/3), int(frame.shape[0]/3)), font, 2, (100, 100, 255), 1, cv2.LINE_AA)

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, "frame:"+str(frame_no), (50, int(frame.shape[0])-50), font, 2, (255, 255, 255), 1, cv2.LINE_AA)

                image = frame

                # Display the resulting frame
                # cv2.imshow('frame', gray)
                cv2.imshow('frame', image)
                out.write(frame)
                cv2.waitKey(1)

                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
        except:
            print("error")
        # When everything done, release the capture
        cap.release()
        out.release()
        cv2.destroyAllWindows()
