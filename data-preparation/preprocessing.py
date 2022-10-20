# -*- coding: utf-8 -*-
import configparser
import os

from samm_data_prepare import sammPreprocessing
from smic_data_prepare import smicPreprocessing
from casmeii_data_prepare import casmeiiPreprocessing
from mmew_data_prepare import mmewPreprocessing

if __name__ == '__main__':
    # 读取配置文件
    config = configparser.ConfigParser()
    config.read("../config.ini")
    # 选择情绪分类方式
    sel_emotions = int(input("请输入情绪分类方式：1<统一数据集分类>, 2<单独数据集分类>\n"))
    # 选择需要进行预处理的数据集
    sel_dataset = input("请输入要进行预处理的数据集编号：1<SAMM>, 2<SMIC>, 3<CASMEII>, 4<MMEW>. "
                "[例如选择SAMM则输入1，若处理多个(如SAMM、SMIC)，则输入12]:\n")

    # 判断进行哪个数据集的预处理
    if '1' in sel_dataset:
        if 'SAMM-'+str(sel_emotions)+'-'+config["IMAGE_PROCESS"]["SAMPLE_NUM"] in os.listdir(config["SAVE_LOC"]["PREPROCESS_SAVE_LOC"]):
            print("SAMM数据集已经进行过相同配置下的预处理，无需再次进行预处理！")
        else:
            print(sammPreprocessing(sel_emotions))

    if '2' in sel_dataset:
        if 'SMIC-'+str(sel_emotions)+'-'+config["IMAGE_PROCESS"]["SAMPLE_NUM"] in os.listdir(config["SAVE_LOC"]["PREPROCESS_SAVE_LOC"]):
            print("SMIC数据集已经进行过相同配置下的预处理，无需再次进行预处理！")
        else:
            print(smicPreprocessing(sel_emotions))

    if '3' in sel_dataset:
        if 'CASMEII-'+str(sel_emotions)+'-'+config["IMAGE_PROCESS"]["SAMPLE_NUM"] in os.listdir(config["SAVE_LOC"]["PREPROCESS_SAVE_LOC"]):
            print("CASMEII数据集已经进行过相同配置下的预处理，无需再次进行预处理！")
        else:
            print(casmeiiPreprocessing(sel_emotions))

    if '4' in sel_dataset:
        if 'MMEW-' + str(sel_emotions) + '-' + config["IMAGE_PROCESS"]["SAMPLE_NUM"] in os.listdir(
                config["SAVE_LOC"]["PREPROCESS_SAVE_LOC"]):
            print("MMEW数据集已经进行过相同配置下的预处理，无需再次进行预处理！")
        else:
            print(mmewPreprocessing(sel_emotions))

    print("Preprocessing done!")

