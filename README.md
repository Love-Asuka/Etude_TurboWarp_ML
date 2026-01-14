# Etude-TurboWarp-ML

一个为[TurboWarp](https://turbowarp.org/)开发的机器学习扩展，允许用户在可视化编程环境中构建和训练神经网络模型。

## 功能特性

### 核心功能
- **神经网络构建**: 支持创建多层神经网络，包括线性层、LayerNorm层、RMSNorm层和独立的激活函数层
- **前向传播**: 实现完整的前向计算流程
- **自动微分**: 支持反向传播算法计算梯度
- **优化器**: 内置SGD和AdamW优化器用于模型训练
- **模型持久化**: 支持以JSON格式保存和加载模型权重

### 支持的激活函数
- ReLU
- Tanh
- Sigmoid
- Softmax
- 无激活函数（线性）

### 支持的损失函数
- 均方误差 (MSE)
- 交叉熵 (Cross Entropy)

### 线性代数工具
- 矩阵乘法
- 矩阵加法

## 使用方法

### 1. 模型定义
使用以下积木块来定义神经网络模型的结构：

```scratch
添加线性层 输入维度 [INPUT_DIM] 输出维度 [OUTPUT_DIM] 使用偏置 [USE_BIAS]
添加激活函数 [ACTIVATION]
添加层归一化 (LayerNorm) 使用偏置 [USE_BIAS]
添加RMS归一化 (RMSNorm)
构建并初始化模型 策略 [INIT]
```

**说明：**
- `添加线性层`：添加一个全连接层，可以指定输入维度、输出维度和是否使用偏置
- `添加激活函数`：添加一个独立的激活函数层，可选择不同的激活函数类型
- `添加层归一化`：添加LayerNorm层，有助于训练稳定性
- `添加RMS归一化`：添加RMSNorm层，是LayerNorm的高效替代方案
- `构建并初始化模型`：完成模型定义并初始化参数，必须在所有层添加完成后调用

### 2. 模型推理
```scratch
推理 输入向量 [INPUT]
```
执行模型的前向传播，得到预测结果。输入向量应为JSON格式的矩阵。

### 3. 模型训练
```scratch
SGD优化 预测 [PRED] 目标 [TARGET] 损失 [LOSS] LR [LR]
AdamW优化 预测 [PRED] 目标 [TARGET] 损失 [LOSS] LR [LR] Decay [DECAY]
```
使用随机梯度下降(SGD)或AdamW算法更新模型参数。需要提供预测值、目标值、损失函数类型和学习率。AdamW优化器还支持权重衰减参数。

### 4. 模型管理
```scratch
导出模型 (JSON)
加载模型 (JSON) [JSON]
清除当前模型
模型已加载?
```
用于模型的保存、加载和状态管理。

### 5. 线性代数运算
```scratch
矩阵 [A] × [B]
矩阵 [A] + [B]
```
提供基本的矩阵运算功能，可用于数据预处理或其他计算。

## 权重文件格式

模型权重以JSON格式存储，包含以下信息：
- 模型元数据（名称、层数、输入/输出维度等）
- 各层参数（权重矩阵和偏置向量）
- 计算图信息

模型格式版本已更新至 1.3，支持独立的激活函数层、LayerNorm 层和 RMSNorm 层。

示例结构：
```json
{
  "format": "etude-ml-model",
  "version": "1.3",
  "meta": {
    "name": "Example-Model",
    "inputDim": 2,
    "outputDim": 1,
    "totalLayers": 4,
    "created": 1700000000000
  },
  "layers": [
    {
      "config": {
        "id": "layer_0_linear",
        "type": "linear",
        "input_dim": 2,
        "output_dim": 2,
        "use_bias": true,
        "input_name": "tensor_0",
        "output_name": "tensor_1"
      },
      "parameters": {
        "weight": [
          [1, 0],
          [0, 1]
        ],
        "bias": [0, 0]
      }
    },
    {
      "config": {
        "id": "layer_1_activation",
        "type": "activation",
        "input_dim": 2,
        "output_dim": 2,
        "activation": "relu",
        "input_name": "tensor_1",
        "output_name": "tensor_2"
      }
    },
    {
      "config": {
        "id": "layer_2_layernorm",
        "type": "layernorm",
        "input_dim": 2,
        "use_bias": true,
        "input_name": "tensor_2",
        "output_name": "tensor_3"
      },
      "parameters": {
        "weight": [1, 1],
        "bias": [0, 0]
      }
    }
  ],
  "computation_graph": {
    "forward": [
      {
        "id": "op_1",
        "type": "linear",
        "inputs": ["tensor_0"],
        "outputs": ["linear_1"],
        "layerId": "layer_0_linear"
      },
      {
        "id": "op_2",
        "type": "activation",
        "activation_type": "relu",
        "inputs": ["linear_1"],
        "outputs": ["tensor_2"]
      },
      {
        "id": "op_3",
        "type": "layernorm",
        "inputs": ["tensor_2"],
        "outputs": ["tensor_3"],
        "layerId": "layer_2_layernorm"
      }
    ],
    "backward": []
  }
}
```

## 技术实现

### 架构设计
项目采用模块化设计，包含以下几个核心类：

1. **EtudeTurboWarpMLCore**: 核心模块，负责模型定义、前向传播和模型管理
2. **EtudeTurboWarpMLAutograd**: 自动微分模块，实现反向传播算法，支持各种层类型的梯度计算
3. **EtudeTurboWarpMLOptimizer**: 优化器模块，提供SGD和AdamW优化算法，用于更新模型参数
4. **EtudeTurboWarpMLLinearAlgebra**: 线性代数工具模块，提供矩阵运算功能

每个模块都实现了特定的功能，并通过代理模式在主类`EtudeTurboWarpML`中统一暴露接口，方便在TurboWarp扩展框架中使用。主类负责协调各个模块的工作，并提供统一的外部接口。


### 计算图
系统通过构建计算图来管理前向和反向传播过程，确保梯度能够正确计算和传播。计算图包含前向传播路径和反向传播路径。

计算图支持多种操作类型，包括线性变换、激活函数层、LayerNorm和RMSNorm等，每种操作都有对应的前向和反向传播实现。现在激活函数作为独立的层存在，使得网络结构更加灵活，可以更容易地在网络中插入或移除激活函数。

### 初始化策略
支持多种参数初始化策略：
- He初始化（默认，适用于ReLU等激活函数）
- Xavier初始化（适用于Sigmoid、Tanh等激活函数）
- 零初始化
- 一初始化

### 优化器
支持两种优化算法：
- **SGD（随机梯度下降）**: 基础的优化算法，通过梯度下降更新模型参数
- **AdamW**: 先进的优化算法，在Adam基础上增加了权重衰减，通常能提供更好的训练效果和收敛速度。支持自定义学习率、β1、β2、epsilon和权重衰减参数。

### 归一化层
支持两种归一化层：
- **LayerNorm（层归一化）**: 对每个样本在特征维度上进行归一化，计算均值和方差并进行标准化
- **RMSNorm（均方根归一化）**: LayerNorm的简化版本，只计算均方根而不计算均值，计算效率更高

两种归一化层都支持前向传播和反向传播的完整实现，可以有效提升模型训练的稳定性和收敛速度。

## 安装和使用

1. 将此扩展加载到TurboWarp中
2. 使用提供的积木块构建神经网络模型
3. 进行模型推理或训练

## 最新进展

- 已完成激活函数独立成层的实现
- AdamW优化器支持更多自定义参数

## 开发计划

- 提供更丰富的损失函数
- 增加更多类型的神经网络层（如卷积层、循环层等）
- 增强错误处理和验证机制
- 添加更多实用工具函数
- 支持批量处理和更复杂的数据格式
- 进一步优化GPU计算后端性能

## 作者

Asuka | BlueIrisSky

## 版本

0.1.0
