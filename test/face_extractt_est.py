
import cv2
import numpy as np

# OpenCV的深度模型人脸检测器，参考：https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html?msclkid=bbba05a1af3911eca0d1cf4ec0faac6c
detector = cv2.FaceDetectorYN.create('/Users/returnyg/PycharmProjects/MER2/utils/face_detection_yunet_2022mar-act_int8-wt_int8-quantized.onnx', "", (224, 224), 0.85, 0.3, 2000)
img = cv2.imread('/Users/returnyg/Datasets/SMIC/SMIC-E_raw image/HS_long/SMIC-HS-E/s03/s3_sur_01/image308866.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
detector.setInputSize((img.shape[1], img.shape[0]))
cv2.imshow('img', img)
cv2.waitKey(0)
# 首次进行人脸检测以便进行配准
faces = detector.detect(img)[-1][0]
faces_range = faces[:-1].astype(np.int32)

face = img[faces_range[1]:faces_range[1] + faces_range[3], faces_range[0]:faces_range[0] + faces_range[2]]
cv2.imshow('face', face)
img = cv2.resize(face, (224, 224))
cv2.imshow('face2', img)
cv2.waitKey(0)

