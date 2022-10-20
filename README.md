# 项目说明 #

- 本项目是由于之前的项目预处理部分过于冗余并且难以修改，加上目前PyTorch框架更加流行，所以决定重构上个项目。
原项目地址：[ReturnYG/MicroExpressionRecognition](https://github.com/ReturnYG/MicroExpressionRecognition "MicroExpressionRecognition")
- 之前的项目没有精力维护，也没有想到有朋友会感兴趣。这次将会花一定时间精力持续完善本项目，欢迎朋友们提交issues。

## 项目结构 ##
- `data-preparation`：数据准备
- `model`：模型
- `utils`：工具
- `train`：训练代码
- `test`：测试项目部分功能代码

## 项目特点 ##
- 通过配置文件使项目更加通用
- 较完整的注释，代码结构清晰，可读性较好
- 模型将采用PyTorch框架(未完成)

## 项目依赖 ##
- Python 3.8
- dlib==19.24.0
- numpy==1.23.4
- opencv_python_inference_engine==4.0.1.2
- pandas==1.5.0

## 项目运行 ##
- 数据准备
    - 申请/下载相关数据集（支持CASME II、SAMM、SMIC、MMEW）
    - 修改`config.ini`文件中的相关路径
    - 运行`data-preparation`下的`preprocessing.py`文件

## 注意事项 ##
- 预处理部分
    - SAMM数据集中的`013_1_12`、`013_1_8`、`017_3_2`、`026_1_1`、`026_1_2`、`035_7_2`样本内图片命名需改成`0xx_0xxx.jpg`。
    - CASMEII数据集中`sub09`下的`EP02_02f`，`sub24`下的`EP02_07`没有提供标签。建议不使用上述样本。

## TODO ##
- [x] 完成基础数据集的预处理部分。
- [ ] 完善预处理部分，增加时间插值法（TIM）、欧拉运动放大（EVM）法、光流特征法等。
- [ ] 增加模型和训练代码。
- [ ] 实现部分先进方法的模型。