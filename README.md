# 全国研究生数学建模竞赛D题目解决方案
##问题描述
+ 问题1. 根据文件“Molecular_Descriptor.xlsx”和“ERα_activity.xlsx”提供的数据，针对1974个化合物的729个分子描述符进行变量选择，根据变量对生物活性影响的重要性进行排序，并给出前20个对生物活性最具有显著影响的分子描述符（即变量），并请详细说明分子描述符筛选过程及其合理性。
+ 问题2. 请结合问题1，选择不超过20个分子描述符变量，构建化合物对ERα生物活性的定量预测模型，请叙述建模过程。然后使用构建的预测模型，对文件“ERα_activity.xlsx”的test表中的50个化合物进行IC50值和对应的pIC50值预测，并将结果分别填入“ERα_activity.xlsx”的test表中的IC50_nM列及对应的pIC50列。
+ 问题3. 请利用文件“Molecular_Descriptor.xlsx”提供的729个分子描述符，针对文件“ADMET.xlsx”中提供的1974个化合物的ADMET数据，分别构建化合物的Caco-2、CYP3A4、hERG、HOB、MN的分类预测模型，并简要叙述建模过程。然后使用所构建的5个分类预测模型，对文件“ADMET.xlsx”的test表中的50个化合物进行相应的预测，并将结果填入“ADMET.xlsx”的test表中对应的Caco-2、CYP3A4、hERG、HOB、MN列。
+ 问题4. 寻找并阐述化合物的哪些分子描述符，以及这些分子描述符在什么取值或者处于什么取值范围时，能够使化合物对抑制ERα具有更好的生物活性，同时具有更好的ADMET性质（给定的五个ADMET性质中，至少三个性质较好）。
## 基本解决思路
+ 灰度模型预测
+ BP、DBN神经网络预测
+ LightGBM/XGBOOST模型进行预测
+ TOPSIS进行预测
## 运行方法
创建虚拟环境
```bash
python -m venv mathmodelenv
source mathmodelenv/bin/activate
```
安装对应的包文件
```bash
pip install -r requirements.txt
```

## 主要问题结果的运行

问题一的求解
```bash
python first.py
```
问题二的求解
```bash
python second.py
```
问题三的求解
```bash
python third.py
```
问题四的部分求解
```bash
python four.py
```

目录下会生成对应的结果文件，文件的位置在`./log`文件夹下面。
