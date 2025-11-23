# Etude-TurboWarp-ML

一个为[TurboWarp](https://turbowarp.org/)开发的机器学习扩展，允许用户在可视化编程环境中构建和训练神经网络模型。

## 功能特性

### 核心功能
- **神经网络构建**: 支持创建多层神经网络，包括线性层和多种激活函数
- **前向传播**: 实现完整的前向计算流程
- **自动微分**: 支持反向传播算法计算梯度
- **优化器**: 内置SGD优化器用于模型训练
- **模型持久化**: 支持以JSON格式保存和加载模型权重

### 支持的激活函数
- ReLU
- Tanh
- Sigmoid
- Softmax
- 无激活函数

### 支持的损失函数
- 均方误差 (MSE)
- 交叉熵 (Cross Entropy)

### 线性代数工具
- 矩阵乘法
- 矩阵加法

## 使用方法

### 1. 模型定义
```scratch
开始模型定义，输入维度 [INPUT_DIM]
添加线性层 输出维度 [OUTPUT_DIM] 激活函数 [ACTIVATION]
结束模型定义
```

### 2. 模型推理
```scratch
推理 输入向量 [INPUT]
```

### 3. 模型训练
```scratch
SGD 预测值 [PRED] 真实值 [TARGET] 损失函数 [LOSS] 学习率 [LR]
```

### 4. 模型管理
```scratch
获取模型结构JSON
清除模型
模型是否已定义
```

## 权重文件格式

模型权重以JSON格式存储，包含以下信息：
- 模型元数据（名称、层数、输入/输出维度等）
- 各层参数（权重矩阵和偏置向量）
- 计算图信息

示例结构：
```json
{
  "format": "turbowarp-nn-weights",
  "version": "1.0",
  "model_meta": {
    "name": "示例神经网络",
    "total_layers": 3,
    "input_dim": 4,
    "output_dim": 2
  },
  "layers": [
    {
      "id": "layer_0_linear",
      "type": "linear",
      "input_dim": 4,
      "output_dim": 8,
      "activation": "relu",
      "parameters": {
        "weight": { "shape": [8, 4], "data": [...] },
        "bias": { "shape": [8], "data": [...] }
      }
    }
  ]
}
```

## 技术实现

### 架构设计
项目采用模块化设计，包含以下几个核心类：

1. **EtudeTurboWarpMLCore**: 核心模块，负责模型定义和前向传播
2. **EtudeTurboWarpMLAutograd**: 自动微分模块，实现反向传播算法
3. **EtudeTurboWarpMLOptimizer**: 优化器模块，提供SGD优化算法
4. **EtudeTurboWarpMLLinearAlgebra**: 线性代数工具模块，提供矩阵运算功能

### 计算图
系统通过构建计算图来管理前向和反向传播过程，确保梯度能够正确计算和传播。

### 初始化策略
支持多种参数初始化策略：
- He初始化
- Xavier初始化
- 零初始化
- 一初始化

## 安装和使用

1. 将此扩展加载到TurboWarp中
2. 使用提供的积木块构建神经网络模型
3. 进行模型推理或训练

## 开发计划

- 支持更多类型的神经网络层
- 增加更多优化算法（如Adam、RMSprop等）
- 提供更丰富的损失函数
- 增强错误处理和验证机制

## 作者

Asuka | Lin Xi

## 版本

0.0.1