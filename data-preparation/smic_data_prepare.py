# -*- coding: utf-8 -*-
import configparser
import os

import numpy as np
import pandas as pd

from utils.image_process import imageProcessing


def smicPreprocessing(sel_emotions):
    # 读取配置文件
    config = configparser.ConfigParser()
    config.read("../config.ini")
    # 读取配置文件中的数据集路径和情绪分类字典
    if sel_emotions == 1:
        EMOTIONS_DICT = eval(config["EMOTIONS_DICT"]["EMOTIONS_DICT_4"])
    else:
        EMOTIONS_DICT = eval(config["EMOTIONS_DICT"]["EMOTIONS_DICT_SMIC"])
    SMIC_LOC = config["DATABASE_LOC"]["SMIC_LOC"]
    SMIC_LABEL_LOC = config["DATABASE_LABEL_LOC"]["SMIC_LABEL_LOC"]
    SAMPLE_NUM = int(config["IMAGE_PROCESS"]["SAMPLE_NUM"])
    PREPROCESS_SAVE_LOC = config["SAVE_LOC"]["PREPROCESS_SAVE_LOC"]

    # 创建最终数据集列表
    smic_data = []
    # 创建受试者编号及绝对路径字典
    sub_dict = {}
    for sub in os.listdir(SMIC_LOC):
        if os.path.isdir(SMIC_LOC + '/' + sub):
            for sample_dir in os.listdir(SMIC_LOC + '/' + sub):
                if os.path.isdir(SMIC_LOC + '/' + sub + '/' + sample_dir):
                    sub_dict[sub + '/' + sample_dir] = SMIC_LOC + '/' + sub + '/' + sample_dir

    # 对字典按照受试者编号排序
    sub_dict = dict(sorted(sub_dict.items(), key=lambda x: x[0]))  # 此处报类型错误，但是不影响运行结果。
    # print(sub_dict)
    sub_num = -1
    index = 0
    for sub, path in sub_dict.items():
        # 此处说明一下期望处理后得到的数据集格式：
        # [ ['sub_num1', [['faces1-数量'], ['labels1-数量']], [['faces2-数量'], ['labels2-数量']] ],
        #   ['sub_num2', [['faces1-数量'], ['labels1-数量']], [['faces2-数量'], ['labels2-数量']] ]
        # ]
        # 目的是方便训练时采取留一法（LOSO）。

        if sub_num == -1:
            sub_num = sub.split('/')[0]
            smic_data.append(sub_num.split(" "))
        elif sub_num != sub.split('/')[0]:
            sub_num = sub.split('/')[0]
            smic_data.append(sub_num.split(" "))
            index += 1
        else:
            pass

        print('Processing sub: ' + sub_num + ', sample num: ' + path.split('/')[-1])
        # 将图像序列路径传递至图像处理函数并返回处理后的图像序列
        onset, apex, offset, label = getSampleInfo(path, SMIC_LABEL_LOC, sel_emotions, EMOTIONS_DICT)
        faces = imageProcessing(path, onset, apex, offset, SAMPLE_NUM, 'SMIC')
        labels = [label] * SAMPLE_NUM
        # 将处理后的图像序列添加至smic_data列表
        sample = [faces, labels]
        smic_data[index].append(sample)
        # print(smic_data[index][0])
        # print(len(smic_data[index][1][0]))
        # print(len(smic_data[index][1][1]))

    print(f"共有{len(smic_data)}个受试者")
    for sub in smic_data:
        print(f"受试者{sub[0]}共有{len(sub[1:])}个样本")
        for index, sample in enumerate(sub[1:]):
            print(f"样本{index+1}共有{len(sample[0])}个图像和{len(sample[1])}个标签")

    # 将预处理好的数据集转为numpy数组保存至本地，以便下次取用
    save_data = np.asarray(smic_data, dtype=object)
    np.save(PREPROCESS_SAVE_LOC + '/' + 'SMIC-' + str(sel_emotions) + '-' + str(SAMPLE_NUM), save_data)

    return 'smic-ok'


def getSampleInfo(path, SMIC_LABEL_LOC, sel_emotions, EMOTIONS_DICT):
    # 读取标签excel文件
    table = pd.read_excel(SMIC_LABEL_LOC, header=0)
    # 获取路径中包含的样本编号
    sub_num = path.split('/')[-1]
    # 获取样本编号对应的信息
    cow = table[table['Filename'] == sub_num]
    # 获取起始帧、顶点帧、终止帧和情绪分类
    onset = int(cow['OnsetF'].values[0])
    offset = int(cow['OffsetF'].values[0])
    apex = (offset + onset) // 2
    emotion = cow['Emotion'].values[0].lower()
    label = EMOTIONS_DICT.get(emotion)

    return onset, apex, offset, label
