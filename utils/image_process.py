# -*- coding: utf-8 -*-
import configparser
import math
import os
import random
import dlib
import numpy as np
import cv2


# 图像预处理主函数，将样本所在文件夹路径下的图像抽取预期的样本数的图像处理为仅包含面部的图像。
def imageProcessing(path, onset, apex, offset, SAMPLE_NUM, dataset_name):
    # 定义需返回的图像序列和标签序列
    faces_list = []

    # 初始图像配准标记点，作用：使面部保持水平并使五官几乎处在不同图像的同一位置
    src_points = -1
    face_range_init = -1
    top, bottom, left, right = 0, 0, 0, 0
    config = configparser.ConfigParser()
    config.read("../config.ini")
    WIDTH = int(config["IMAGE_PROCESS"]["WIDTH"])
    HEIGHT = int(config["IMAGE_PROCESS"]["HEIGHT"])
    FACE_DETECTION_MODEL = config["EXTRA_FILE_LOC"]["FACE_DETECTION_MODEL"]
    DLIB_FACE_DETECTION_MODEL = config["EXTRA_FILE_LOC"]["DLIB_FACE_DETECTION_MODEL"]

    # OpenCV的深度模型人脸检测器，参考：https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html?msclkid=bbba05a1af3911eca0d1cf4ec0faac6c
    # SMIC部分数据集在阈值设定为0.9时会出现人脸检测不到的情况，因此将阈值设定为0.85
    if dataset_name == "SMIC":
        detector = cv2.FaceDetectorYN.create(FACE_DETECTION_MODEL, "", (224, 224), 0.8, 0.3, 2000)
    else:
        detector = cv2.FaceDetectorYN.create(FACE_DETECTION_MODEL, "", (WIDTH, HEIGHT), 0.9, 0.3, 5000)
    # print(sorted(os.listdir(path)))

    # 根据给定的样本数量和始末帧调整图像数量
    img_list = selectImg(path, onset, apex, offset, SAMPLE_NUM, dataset_name)
    # print(img_list)

    for image in img_list:
        img = cv2.imread(path + '/' + image)
        # 每个样本第一张图像确定面部区域，之后将一直利用此区域剪裁图像，目的是保证图像稳定，不抖动
        if face_range_init == -1 and dataset_name != "MMEW":
            # 由于部分样本确定的面部区域不准确，因此通过随机抽取一张图片的方式确定样本的面部区域
            rand_img = random.choice(img_list)
            img_rand = cv2.imread(path + '/' + rand_img)
            print("随机抽取的图像为：", rand_img, ", 路径为：", path + '/' + rand_img)
            detector.setInputSize((img_rand.shape[1], img_rand.shape[0]))

            # 人脸检测并确定面部区域
            faces = detector.detect(img_rand)[-1][0]
            faces_range = faces[:-1].astype(np.int32)

            # 根据需求决定是否进行配准，若不进行配准则直接注释下面的代码块
            if config["IMAGE_PROCESS"]["IMG_REGISTRATION"] == 'True':
                # 以第一张图像为基准进行面部配准，记录配准点
                if type(src_points) == int:
                    img_rand, src_points = faceRegistration(img_rand, faces_range)
                else:
                    img_rand = faceRegistration(img_rand, faces_range, src_points)
                # 配准后再进行人脸检测以获取新的面部区域位置
                faces = detector.detect(img_rand)[-1][0]
                faces_range = faces[:-1].astype(np.int32)

            top_opencv = faces_range[1]
            bottom_opencv = faces_range[1] + faces_range[3]
            left_opencv = faces_range[0]
            right_opencv = faces_range[0] + faces_range[2]
            face_range_init = 0

            # 为确保每张图像的面部区域都是完整的，在这里增加Dlib的人脸检测器进行校验，
            # 如果检测到的面部区域不完整，则使用Dlib的人脸检测器进行校正。会增加预处理的开销，可选择注释掉。
            # face = img[top_opencv:bottom_opencv, left_opencv:right_opencv]
            # cv2.imshow('opencv', face)
            top_dlib, bottom_dlib, left_dlib, right_dlib = verifyFacialRegion(path + '/' + rand_img, faces_range, DLIB_FACE_DETECTION_MODEL)
            # face2 = img[top_dlib:bottom_dlib, left_dlib:right_dlib]
            # cv2.imshow('dlib', face2)
            # face3 = img[min(top_opencv, top_dlib):max(bottom_opencv, bottom_dlib), min(left_opencv, left_dlib):max(right_opencv, right_dlib)+20]
            # cv2.imshow('merge', face3)
            # cv2.waitKey(1)

            # 经过测试，结合两种人脸检测器的结果，可以得到更加完整的面部区域
            top = min(top_opencv, top_dlib)
            bottom = max(bottom_opencv, bottom_dlib)
            left = min(left_opencv, left_dlib)
            right = max(right_opencv, right_dlib)+20  # 两种检测器识别的面部区域的右侧有一定的偏差，因此这里增加20个像素点的偏差

        else:
            # MMEW样本已剪裁完成
            top = 0
            bottom = img.shape[0]
            left = 0
            right = img.shape[1]  # 两种检测器识别的面部区域的右侧有一定的偏差，因此这里增加20个像素点的偏差

        # 根据面部区域剪裁图像face = img[上:下, 左:右]
        face = img[top:bottom, left:right]
        # cv2.imshow('before', face)
        # cv2.waitKey(1)
        # print(path + '/' + image)
        img = cv2.resize(face, (WIDTH, HEIGHT))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # print(img.shape)
        cv2.imshow('final', img)
        cv2.waitKey(1)
        faces_list.append(img)
        # print(img.shape)

    return faces_list


def faceRegistration(img, faces_range, src_points=None):
    # 面部配准，若未设置基准点，则将面部旋转至水平并返回计算基准点，若已设置基准点，则通过基准点进行配准并返回配准后的图像
    if src_points is None:
        # cv2.imshow('frist-raw', img)

        # 首次进行面部配准，需要再修改后重新定位面部标记点，因此需要检测器
        detector = cv2.FaceDetectorYN.create('./face_detection_yunet_2022mar.onnx', "", (img.shape[1], img.shape[0]), 0.9, 0.3, 5000)

        # 获取左右眼的坐标
        left_eye = [int(faces_range[6]), int(faces_range[7])]
        right_eye = [int(faces_range[4]), int(faces_range[5])]

        # 计算眼睛质心之间的角度
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]

        # 计算两个中心线与水平线之间的角度
        angle = math.atan2(dy, dx) * 180. / math.pi

        # 计算两眼的中心
        eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

        # 在两眼的中心处，按角度旋转图像
        rotate_matrix = cv2.getRotationMatrix2D(eye_center, angle-180, scale=1)
        rotated_img = cv2.warpAffine(img, rotate_matrix, (img.shape[1], img.shape[0]))

        # 计算新的标记点位置并返回以便后续图像配准，此处代码个人觉得不够好，但是一时没有想出来怎么再优化，欢迎大家提出意见
        faces = detector.detect(rotated_img)[-1][0]
        faces_range = faces[:-1].astype(np.int32)

        # 新的标记点并返回-透视变换坐标（需要四个点）
        src_points = np.array([[faces_range[6], faces_range[7]], [faces_range[4], faces_range[5]],
                               [faces_range[10], faces_range[11]], [faces_range[12], faces_range[13]]], dtype=np.float32)
        # 获取需要配准的图像的标记点-透视变换坐标（需要三个点）
        # src_points = np.array([[faces_range[6], faces_range[7]], [faces_range[4], faces_range[5]],
        #                        [faces_range[8], faces_range[9]]], dtype=np.float32)

        # cv2.imshow('frist-after', rotated_img)
        # cv2.waitKey(0)
        return rotated_img, src_points
    else:
        # cv2.imshow('continue-raw', img)
        # 获取需要配准的图像的标记点-透视变换坐标（需要四个点）
        den_points = np.array([[faces_range[6], faces_range[7]], [faces_range[4], faces_range[5]],
                               [faces_range[10], faces_range[11]], [faces_range[12], faces_range[13]]], dtype=np.float32)
        # 获取需要配准的图像的标记点-透视变换坐标（需要三个点）
        # den_points = np.array([[faces_range[6], faces_range[7]], [faces_range[4], faces_range[5]],
        #                        [faces_range[8], faces_range[9]]], dtype=np.float32)

        # 透视变化-计算变换矩阵
        T = cv2.getPerspectiveTransform(src_points, den_points)
        # 进行透视变换
        warp_imgae = cv2.warpPerspective(img, T, (img.shape[1], img.shape[0]))

        # 仿射变换-计算变换矩阵
        # M = cv2.getAffineTransform(src_points, den_points)
        # 进行仿射变换
        # warp_imgae = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        # cv2.imshow('continue-after', warp_imgae)
        # cv2.waitKey(0)
        return warp_imgae


def selectImg(path, onset, apex, offset, SAMPLE_NUM, dataset_name):
    # 该函数通过计算始末帧数量和样本数之间的差，动态调整样本图像的文件
    img_list = []

    # CASMEII有1个样本未设定APEX帧，则取其一半位置为APEX帧
    if apex == '/':
        apex = (onset + offset) // 2
    # 1、先计算始末帧数量判断增加帧还是减少帧
    oringin_num = offset - onset + 1
    # 2、计算差值
    diff_num = oringin_num - SAMPLE_NUM
    apex_repeat = 0
    # 3、判断增加帧还是减少帧
    if diff_num >= 0:
        # 3.1、以apex帧为中心减少帧
        first_half = apex - onset + 1
        second_half = offset - apex + 1
        while diff_num > 0:
            if first_half > second_half:
                onset += 1
            else:
                offset -= 1
            diff_num -= 1

        # 由于数据集命名不统一，因此在此处进行判断
        # SAMM
        if dataset_name == 'SAMM':
            conn = '_0' if '_0' in os.listdir(path)[0] else '_'
            for i in range(onset, offset + 1):
                img_list.append(path.split('/')[-2] + conn + str(i) + '.jpg')
        # CASMEII
        elif dataset_name == 'CASMEII':
            conn = 'img'
            for i in range(onset, offset + 1):
                img_list.append(conn + str(i) + '.jpg')
        # SMIC
        elif dataset_name == 'SMIC':
            # 命名规则原因加此判断
            if offset < 100000:
                conn = 'image0'
            else:
                conn = 'image'
            for i in range(onset, offset + 1):
                img_list.append(conn + str(i) + '.jpg')
        # MMEW
        elif dataset_name == 'MMEW':
            conn = ''
            for i in range(onset, offset + 1):
                img_list.append(conn + str(i) + '.jpg')

    elif diff_num < 0:
        # 3.2、以apex帧为中心增加帧
        while diff_num < 0:
            apex_repeat += 1
            diff_num += 1

        # 由于数据集命名不统一，因此在此处进行判断
        # SAMM
        if dataset_name == 'SAMM':
            conn = '_0' if '_0' in os.listdir(path)[0] else '_'
            for i in range(onset, apex + 1):
                img_list.append(path.split('/')[-2] + conn + str(i) + '.jpg')
            for i in range(apex_repeat):
                img_list.append(path.split('/')[-2] + conn + str(apex) + '.jpg')
            for i in range(apex + 1, offset + 1):
                img_list.append(path.split('/')[-2] + conn + str(i) + '.jpg')
        elif dataset_name == 'CASMEII':
            conn = 'img'
            for i in range(onset, apex + 1):
                img_list.append(conn + str(i) + '.jpg')
            for i in range(apex_repeat):
                img_list.append(conn + str(apex) + '.jpg')
            for i in range(apex + 1, offset + 1):
                img_list.append(conn + str(i) + '.jpg')
        elif dataset_name == 'SMIC':
            # 命名规则原因加此判断
            if offset < 100000:
                conn = 'image0'
            else:
                conn = 'image'
            for i in range(onset, apex + 1):
                img_list.append(conn + str(i) + '.jpg')
            for i in range(apex_repeat):
                img_list.append(conn + str(apex) + '.jpg')
            for i in range(apex + 1, offset + 1):
                img_list.append(conn + str(i) + '.jpg')
        elif dataset_name == 'MMEW':
            conn = ''
            for i in range(onset, apex + 1):
                img_list.append(conn + str(i) + '.jpg')
            for i in range(apex_repeat):
                img_list.append(conn + str(apex) + '.jpg')
            for i in range(apex + 1, offset + 1):
                img_list.append(conn + str(i) + '.jpg')

    print('onset:{}, offset:{}, apex:{}'.format(onset, offset, apex))
    print('img_list:', img_list)
    print(len(img_list))

    return img_list


def verifyFacialRegion(rand_img, faces_range, DLIB_FACE_DETECTION_MODEL):
    cnn_face_detector = dlib.cnn_face_detection_model_v1(DLIB_FACE_DETECTION_MODEL)
    # 1、读取图像文件
    img = dlib.load_rgb_image(rand_img)
    # 2、检测人脸
    dets = cnn_face_detector(img, 1)
    return dets[0].rect.top(), dets[0].rect.bottom(), dets[0].rect.left(), dets[0].rect.right()
