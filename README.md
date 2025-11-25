# Etude-TurboWarp-ML

一个为[TurboWarp](https://turbowarp.org/)开发的机器学习扩展，允许用户在可视化编程环境中构建和训练神经网络模型。

## 功能特性

### 核心功能
- **神经网络构建**: 支持创建多层神经网络，包括线性层、LayerNorm层和RMSNorm层，以及多种激活函数
- **前向传播**: 实现完整的前向计算流程
- **自动微分**: 支持反向传播算法计算梯度
- **优化器**: 内置SGD优化器用于模型训练
- **模型持久化**: 支持以JSON格式保存和加载模型权重

### 支持的激活函数
- ReLU
- Tanh
- Sigmoid
- Softmax

### 支持的损失函数
- 均方误差 (MSE)
- 交叉熵 (Cross Entropy)

### 线性代数工具
- 矩阵乘法
- 矩阵加法

## 使用方法

### 1. 模型定义
```scratch
添加线性层 输出维度 [OUTPUT_DIM] 激活函数 [ACTIVATION] 使用偏置 [USE_BIAS]
添加层归一化 (LayerNorm) 使用偏置 [USE_BIAS]
添加RMS归一化 (RMSNorm)
构建并初始化模型 输入维度 [INPUT_DIM] 策略 [INIT]
```

### 2. 模型推理
```scratch
推理 输入向量 [INPUT]
```

### 3. 模型训练
```scratch
SGD优化 预测 [PRED] 目标 [TARGET] 损失 [LOSS] LR [LR]
```

### 4. 模型管理
```scratch
导出模型 (JSON)
加载模型 (JSON) [JSON]
清除当前模型
模型已加载?
```

### 5. 线性代数运算
```scratch
矩阵 [A] × [B]
矩阵 [A] + [B]
```

## 权重文件格式

模型权重以JSON格式存储，包含以下信息：
- 模型元数据（名称、层数、输入/输出维度等）
- 各层参数（权重矩阵和偏置向量）
- 计算图信息

模型格式版本已更新至 1.1，支持 LayerNorm 和 RMSNorm 层。

示例结构：
```json
{
  "format": "etude-ml-model",
  "version": "1.1",
  "meta": {
    "name": "Example-Model",
    "inputDim": 2,
    "outputDim": 1,
    "totalLayers": 3,
    "created": 1700000000000
  },
  "layers": [
    {
      "config": {
        "id": "layer_0_linear",
        "type": "linear",
        "input_dim": 2,
        "output_dim": 2,
        "activation": "relu",
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
        "id": "layer_1_layernorm",
        "type": "layernorm",
        "input_dim": 2,
        "use_bias": true,
        "input_name": "tensor_1",
        "output_name": "tensor_2"
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
        "type": "layernorm",
        "inputs": ["linear_1"],
        "outputs": ["tensor_2"],
        "layerId": "layer_1_layernorm"
      }
    ],
    "backward": []
  }
}
```

## 技术实现

### 架构设计
项目采用模块化设计，包含以下几个核心类：

1. **EtudeTurboWarpMLCore**: 核心模块，负责模型定义和前向传播
2. **EtudeTurboWarpMLAutograd**: 自动微分模块，实现反向传播算法
3. **EtudeTurboWarpMLOptimizer**: 优化器模块，提供SGD优化算法
4. **EtudeTurboWarpMLLinearAlgebra**: 线性代数工具模块，提供矩阵运算功能

每个模块都实现了特定的功能，并通过代理模式在主类中统一暴露接口，方便在TurboWarp扩展框架中使用。

### 计算图
系统通过构建计算图来管理前向和反向传播过程，确保梯度能够正确计算和传播。计算图包含前向传播路径和反向传播路径。

### 初始化策略
支持多种参数初始化策略：
- He初始化（默认）
- Xavier初始化
- 零初始化
- 一初始化

### 归一化层
支持两种归一化层：
- **LayerNorm（层归一化）**: 对每个样本在特征维度上进行归一化
- **RMSNorm（均方根归一化）**: LayerNorm的简化版本，计算效率更高

## 安装和使用

1. 将此扩展加载到TurboWarp中
2. 使用提供的积木块构建神经网络模型
3. 进行模型推理或训练

## 开发计划

- 增加更多优化算法（如Adam、RMSprop等）
- 提供更丰富的损失函数
- 增加更多类型的神经网络层（如卷积层、循环层等）
- 增强错误处理和验证机制
- 添加更多实用工具函数
- 支持批量处理和更复杂的数据格式

## 作者

Asuka | Lin Xi

## 版本

0.0.5