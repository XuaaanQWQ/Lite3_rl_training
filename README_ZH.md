# RL-Lite3

[English](./README.md)

## 简介
一种基于学习的四足机器人运动控制器。包含在云深处科技绝影Lite3上进行强化学习训练和硬件部署所需的所有组件。
## 软件架构
本仓库由包含以下目录：
- rsl_rl: 一个封装了强化学习方法的包。
- legged_gym: 基于 Gym 环境、专为四足机器人设计的仿真框架。


## 准备环境 
1.  在Ubuntu系统中创建一个python（3.6/3.7/3.8，建议使用3.8）环境。

2.  安装计算平台为CUDA的PyTorch。
```
# pytorch
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

3.  从官方网站下载[Isaac Gym](https://developer.nvidia.com/isaac-gym)（版本 >= preview 3），并将其放入项目的根目录中。

4. 使用`pip`安装python依赖项。
```
pip3 install transformations matplotlib gym tensorboard numpy=1.23.5
```

5. 通过 pip 安装 legged_gym 和 rsl_rl
```
cd legged_gym
pip install -e .

cd rsl_rl
pip install -e .
```

# 使用方法

### 在仿真环境中训练策略
```
cd ${PROJECT_DIR}
python3 legged_gym/legged_gym/scripts/train.py --rl_device cuda:0 --sim_device cuda:0 --headless
```

### 在仿真环境中运行控制器
```
cd ${PROJECT_DIR}
python3 legged_gym/legged_gym/scripts/play.py --rl_device cuda:0 --sim_device cuda:0 --load_run ${model_dir} --checkpoint ${model_name}
```
检查您的计算机是否有GPU，若无，请将上述脚本中的单词 `cuda:0` 替换为 `cpu`。
通过 `--load_run` 和 `--checkpoint`  指定网络模型的路径。

### 在现实环境中运行控制器

将策略文件复制到项目[rl_deploy](https://github.com/DeepRoboticsLab/Lite3_rl_deploy.git)中,然后，您可以在现实环境中运行强化学习控制器

## 参考资料
- [legged_gym](https://github.com/leggedrobotics/legged_gym.git)
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl)
- [quadruped-robot](https://gitee.com/HUAWEI-ASCEND/quadruped-robot.git)
  
  
  
[联系我们](https://www.deeprobotics.cn/robot/index/company.html#maps)

