const MLUtils = {
  transpose(matrix) {
    if (!matrix || !Array.isArray(matrix) || matrix.length === 0 || !Array.isArray(matrix[0])) return [];
    return matrix[0].map((_, col) => matrix.map(row => row[col]));
  },

  matMul(a, b) {
    if (!a || !b || a.length === 0 || b.length === 0 || !a[0] || !b[0] || a[0].length !== b.length) {
      if (a && b && a[0] && b[0] && a[0].length !== b.length) {
        console.error(`[ML] 矩阵乘法维度不匹配: A[0].length=${a[0].length} ≠ B.length=${b.length}`);
      }
      return [];
    }
    return a.map(row => 
      b[0].map((_, j) => 
        row.reduce((sum, val, k) => sum + val * b[k][j], 0)
      )
    );
  },

  sumRows(matrix) {
    if (!matrix || !matrix[0]) return [];
    return matrix[0].map((_, col) => 
      matrix.reduce((sum, row) => sum + row[col], 0)
    );
  },

  hadamard(a, b) {
    if (!a || !b || a.length !== b.length || a[0].length !== b[0].length) return [];
    return a.map((row, i) => row.map((val, j) => val * b[i][j]));
  },

  applyActivation(matrix, type) {
    if (!matrix) return [];
    switch(type) {
      case 'relu':
        return matrix.map(row => row.map(val => Math.max(0, val)));
      case 'tanh':
        return matrix.map(row => row.map(val => Math.tanh(val)));
      case 'sigmoid':
        return matrix.map(row => row.map(val => 1 / (1 + Math.exp(-val))));
      case 'softmax':
        return matrix.map(row => {
          const maxVal = Math.max(...row);
          const exps = row.map(val => Math.exp(val - maxVal));
          const sumExps = exps.reduce((a, b) => a + b, 0);
          return exps.map(exp => exp / sumExps);
        });
      default:
        return matrix;
    }
  },

  activationDerivative(postActivation, activation_type, outputGrad) {
    if (!postActivation || !outputGrad) return [];
    
    switch(activation_type) {
      case 'relu':
        const reluGrad = postActivation.map(row => row.map(val => val > 0 ? 1 : 0));
        return MLUtils.hadamard(reluGrad, outputGrad);
      
      case 'tanh':
        const tanhGrad = postActivation.map(row => row.map(val => 1 - val * val));
        return MLUtils.hadamard(tanhGrad, outputGrad);
      
      case 'sigmoid':
        const sigGrad = postActivation.map(row => row.map(val => val * (1 - val)));
        return MLUtils.hadamard(sigGrad, outputGrad);
      
      case 'softmax':
        return postActivation.map((row, i) => {
          const rowGrad = outputGrad[i];
          const n = row.length;
          const result = new Array(n).fill(0);
          
          for (let j = 0; j < n; j++) {
            for (let k = 0; k < n; k++) {
              const jacobian = j === k ? row[k] * (1 - row[k]) : -row[j] * row[k];
              result[j] += rowGrad[k] * jacobian;
            }
          }
          return result;
        });
      
      default:
        return outputGrad.map(row => [...row]);
    }
  },

  computeLossAndGradient(pred, target, lossType) {
    if (!pred || !target || pred.length !== target.length) {
      console.error('[ML] 预测值与真实值维度不匹配');
      return { loss: 0, grad: [] };
    }

    switch(lossType) {
      case 'mse':
        const batchSize = pred.length;
        const mseGrad = pred.map((row, i) => 
          row.map((val, j) => 2 * (val - target[i][j]) / batchSize)
        );
        const mseLoss = pred.reduce((sum, row, i) => 
          sum + row.reduce((rowSum, val, j) => rowSum + Math.pow(val - target[i][j], 2), 0), 0
        ) / batchSize;
        return { loss: mseLoss, grad: mseGrad };

      case 'crossentropy':
        const epsilon = 1e-7;
        const ceLoss = -pred.reduce((sum, row, i) => 
          sum + row.reduce((rowSum, val, j) => rowSum + target[i][j] * Math.log(val + epsilon), 0), 0
        ) / pred.length;

        const ceGrad = pred.map((row, i) => 
          row.map((val, j) => val - target[i][j])
        );
        return { loss: ceLoss, grad: ceGrad };

      default:
        console.error(`[ML] 不支持的损失函数: ${lossType}`);
        return { loss: 0, grad: [] };
    }
  }
};

class EtudeTurboWarpMLCore {
  constructor() {
    this.globalState = this._createFreshState();
  }

  _createFreshState() {
    return {
      layers: [],
      currentInputDim: null,
      isModelDefined: false,
      computationGraph: { forward: [], backward: [] },
      modelMeta: {
        name: '未命名模型',
        inputDim: null,
        outputDim: null,
        totalLayers: 0,
        created: null
      },
      parameters: {},
      gradients: {},
      forwardData: {} 
    };
  }

  startModelDefinition(args) {
    const inputDim = parseInt(args.INPUT_DIM);
    if (isNaN(inputDim) || inputDim <= 0) {
      console.error('[core] 输入维度必须是正整数');
      return;
    }

    this.globalState = this._createFreshState();
    this.globalState.currentInputDim = inputDim;
    this.globalState.modelMeta.inputDim = inputDim;
    console.log(`[core] 模型定义开始，输入维度: ${inputDim}`);
  }

  addLinearLayer(args) {
    if (this.globalState.currentInputDim === null) {
      console.error('[core] 请先调用"开始模型定义"');
      return;
    }

    const outputDim = parseInt(args.OUTPUT_DIM);
    const activation = args.ACTIVATION;
    if (isNaN(outputDim) || outputDim <= 0) {
      console.error('[core] 输出维度必须是正整数');
      return;
    }

    const layerIndex = this.globalState.layers.length;
    const layerId = `layer_${layerIndex}_linear`;
    
    const inputName = layerIndex === 0 ? 'tensor_0' : `tensor_${layerIndex}`;
    const linearOutputName = `linear_${layerIndex + 1}`;
    const activationOutputName = `tensor_${layerIndex + 1}`;

    const layerConfig = {
      id: layerId,
      type: 'linear',
      input_dim: this.globalState.currentInputDim,
      output_dim: outputDim,
      activation: activation,
      input_name: inputName,
      output_name: activationOutputName
    };

    this.globalState.layers.push(layerConfig);

    this.globalState.computationGraph.forward.push({
      id: `op_${layerIndex + 1}`,
      type: 'linear',
      inputs: [inputName],
      outputs: [linearOutputName],
      params: [`${layerId}.weight`, `${layerId}.bias`]
    });

    if (activation !== 'none') {
      this.globalState.computationGraph.forward.push({
        id: `act_${layerIndex + 1}`,
        type: 'activation',
        activation_type: activation,
        inputs: [linearOutputName],
        outputs: [activationOutputName]
      });
    }

    this.globalState.gradients[layerId] = {
      weight: Array(outputDim).fill().map(() => Array(this.globalState.currentInputDim).fill(0)),
      bias: Array(outputDim).fill(0)
    };

    this.globalState.currentInputDim = outputDim;
    this.globalState.modelMeta.totalLayers = this.globalState.layers.length;
    this.globalState.modelMeta.outputDim = outputDim;

    console.log(`[core] 添加层 ${layerIndex}: ${layerId} (${inputName} -> ${activationOutputName})`);
  }

  endModelDefinition() {
    if (this.globalState.layers.length === 0) {
      console.error('[core] 模型中至少需要一个层');
      return;
    }

    this.globalState.isModelDefined = true;

    // He初始化
    this.globalState.layers.forEach((layer) => {
      const inDim = layer.input_dim;
      const outDim = layer.output_dim;
      const layerId = layer.id;
      const scale = Math.sqrt(2.0 / inDim);
      
      this.globalState.parameters[layerId] = {
        weight: Array(outDim).fill().map(() => 
          Array(inDim).fill().map(() => (Math.random() - 0.5) * 2 * scale)
        ),
        bias: Array(outDim).fill(0)
      };
    });

    console.log('[core] 模型定义完成，参数已初始化');
  }

  forward(args) {
    if (!this.globalState.isModelDefined) {
      console.error('[core] 模型未定义，无法执行推理');
      return '[]';
    }

    let input;
    try {
      input = JSON.parse(args.INPUT);
    } catch(e) {
      console.error('[core] 输入格式无效，需为JSON数组');
      return '[]';
    }

    if (!Array.isArray(input) || !Array.isArray(input[0])) {
      console.error('[core] 输入必须是二维数组 [batch, features]');
      return '[]';
    }

    const expectedDim = this.globalState.modelMeta.inputDim;
    if (input[0].length !== expectedDim) {
      console.error(`[core] 输入特征维度不匹配: 期望 ${expectedDim}, 得到 ${input[0].length}`);
      return '[]';
    }

    this.globalState.forwardData = {};
    this.globalState.forwardData['tensor_0'] = {
      preActivation: null, 
      postActivation: input 
    };
    let currentTensor = input;
    let currentPreActivation = null;

    for (const node of this.globalState.computationGraph.forward) {
      if (node.type === 'linear') {
        currentTensor = this._linearForward(node, currentTensor);
        currentPreActivation = currentTensor; // 线性层输出是"激活前"
      } else if (node.type === 'activation') {
        const activatedTensor = MLUtils.applyActivation(currentTensor, node.activation_type);

        this.globalState.forwardData[node.outputs[0]] = {
          preActivation: currentTensor,
          postActivation: activatedTensor
        };
        
        currentTensor = activatedTensor;
        currentPreActivation = null; // 重置
      }
    }

    return JSON.stringify(currentTensor);
  }

  _linearForward(node, input) {
    const layerId = node.params[0].split('.')[0];
    const layerParams = this.globalState.parameters[layerId];
    
    if (!layerParams) {
      console.error(`[core] 未找到层 ${layerId} 的参数`);
      return input;
    }

    const weightT = MLUtils.transpose(layerParams.weight);
    const output = MLUtils.matMul(input, weightT);
    
    return output.map(row => row.map((val, j) => val + layerParams.bias[j]));
  }

  getModelStructure() {
    if (!this.globalState.isModelDefined) {
      return JSON.stringify({ error: '模型尚未定义' }, null, 2);
    }

    return JSON.stringify({
      format: 'turbowarp-nn-weights',
      version: '1.0',
      model_meta: this.globalState.modelMeta,
      layers: this.globalState.layers.map(layer => ({
        ...layer,
        parameters: {
          weight: {
            shape: [layer.output_dim, layer.input_dim],
            data: this.globalState.parameters[layer.id]?.weight || []
          },
          bias: {
            shape: [layer.output_dim],
            data: this.globalState.parameters[layer.id]?.bias || []
          }
        }
      })),
      computation_graph: this.globalState.computationGraph
    }, null, 2);
  }

  clearModel() {
    this.globalState = this._createFreshState();
    console.log('[core] 模型已清除');
  }

  isModelDefined() {
    return this.globalState.isModelDefined;
  }
}

class EtudeTurboWarpMLAutograd {
  constructor(coreInstance) {
    this.core = coreInstance;
  }

  backward(args) {
    if (!this.core.globalState.isModelDefined) {
      console.error('[autograd] 模型未定义');
      return;
    }

    let outputGrad;
    try {
      outputGrad = JSON.parse(args.GRAD);
    } catch(e) {
      console.error('[autograd] 梯度格式错误，需为JSON数组');
      return;
    }

    const tape = [...this.core.globalState.computationGraph.forward].reverse();
    const gradBuffer = {};

    const firstNode = tape[0];
    if (!firstNode) {
      console.error('[autograd] 计算图为空');
      return;
    }
    
    gradBuffer[firstNode.inputs[0]] = outputGrad;

    for (const node of tape) {
      if (node.type === 'linear') {
        this._linearBackward(node, gradBuffer);
      } else if (node.type === 'activation') {
        this._activationBackward(node, gradBuffer);
      }
    }
    
    console.log('[autograd] 反向传播完成');
  }

  _linearBackward(node, gradBuffer) {
    const inputName = node.inputs[0];
    const outputName = node.outputs[0];
    const outputGrad = gradBuffer[outputName];
    
    if (!outputGrad) {
      console.warn(`[autograd] 未找到 ${outputName} 的梯度`);
      return;
    }

    const layerId = node.params[0].split('.')[0];
    const layer = this.core.globalState.layers.find(l => l.id === layerId);
    const inputData = this.core.globalState.forwardData?.[inputName]?.postActivation;
    const weight = this.core.globalState.parameters[layerId]?.weight;
    
    if (!layer || !inputData || !weight) {
      console.error(`[autograd] 缺少线性层 ${layerId} 所需数据`);
      return;
    }

    const weightGrad = MLUtils.matMul(MLUtils.transpose(outputGrad), inputData);
    const biasGrad = MLUtils.sumRows(outputGrad);
    const inputGrad = MLUtils.matMul(outputGrad, MLUtils.transpose(weight));

    gradBuffer[inputName] = inputGrad;
    this.core.globalState.gradients[layerId] = { 
      weight: weightGrad, 
      bias: biasGrad 
    };
  }

  _activationBackward(node, gradBuffer) {
    const inputName = node.inputs[0];
    const outputName = node.outputs[0];
    const outputGrad = gradBuffer[outputName];
    
    if (!outputGrad) {
      console.warn(`[autograd] 未找到 ${outputName} 的梯度`);
      return;
    }

    const activationData = this.core.globalState.forwardData?.[outputName];
    if (!activationData) {
      console.error(`[autograd] 未找到激活层数据 ${outputName}`);
      return;
    }

    const inputGrad = MLUtils.activationDerivative(
      activationData.postActivation, 
      node.activation_type, 
      outputGrad
    );
    
    gradBuffer[inputName] = inputGrad;
  }

  zeroGrad() {
    this.core.globalState.layers.forEach(layer => {
      const { id, input_dim: inDim, output_dim: outDim } = layer;
      this.core.globalState.gradients[id] = {
        weight: Array(outDim).fill().map(() => Array(inDim).fill(0)),
        bias: Array(outDim).fill(0)
      };
    });
    console.log('[autograd] 梯度已清零');
  }
}

class EtudeTurboWarpMLOptimizer {
  constructor(coreInstance) {
    this.core = coreInstance;
  }

  stepSGD(args) {
    if (!this.core.globalState.isModelDefined) {
      console.error('[optimizer] 模型未定义');
      return;
    }

    let pred, target;
    try {
      pred = JSON.parse(args.PRED);
      target = JSON.parse(args.TARGET);
    } catch(e) {
      console.error('[optimizer] 预测值或真实值格式错误，需为JSON数组');
      return;
    }

    const lossType = args.LOSS || 'mse';
    const learningRate = parseFloat(args.LR) || 0.01;

    this.core.autograd.zeroGrad();

    const { loss, grad } = MLUtils.computeLossAndGradient(pred, target, lossType);
    console.log(`[optimizer] 损失值 (${lossType}): ${loss.toFixed(6)}`);

    this.core.autograd.backward({ GRAD: JSON.stringify(grad) });

    let updateCount = 0;
    this.core.globalState.layers.forEach(layer => {
      const layerId = layer.id;
      const layerGrad = this.core.globalState.gradients[layerId];
      const layerParams = this.core.globalState.parameters[layerId];
      
      if (!layerGrad || !layerParams) {
        console.warn(`[optimizer] 跳过未初始化的层 ${layerId}`);
        return;
      }

      layerParams.weight = layerParams.weight.map((row, i) =>
        row.map((val, j) => val - learningRate * layerGrad.weight[i][j])
      );
      
      layerParams.bias = layerParams.bias.map((val, i) =>
        val - learningRate * layerGrad.bias[i]
      );
      
      updateCount++;
    });

    console.log(`[optimizer] SGD完成，更新 ${updateCount} 层，学习率: ${learningRate}`);
  }
}

class EtudeTurboWarpMLLinearAlgebra {
  matrix_multiplication(args) {
    try {
      const a = JSON.parse(args.A);
      const b = JSON.parse(args.B);
      const result = MLUtils.matMul(a, b);
      return JSON.stringify(result);
    } catch(e) {
      return '[]';
    }
  }

  matrix_add(args) {
    try {
      const a = JSON.parse(args.A);
      const b = JSON.parse(args.B);
      if (!a || !b || a.length !== b.length || !a[0] || !b[0] || a[0].length !== b[0].length) {
        return '[]';
      }
      const result = a.map((row, i) => row.map((val, j) => val + b[i][j]));
      return JSON.stringify(result);
    } catch(e) {
      return '[]';
    }
  }
}

class EtudeTurboWarpML {
  constructor() {
    this.core = new EtudeTurboWarpMLCore();
    this.autograd = new EtudeTurboWarpMLAutograd(this.core);
    this.optimizer = new EtudeTurboWarpMLOptimizer(this.core);
    this.linearAlgebra = new EtudeTurboWarpMLLinearAlgebra();
  }

  getInfo() {
    return {
      id: 'EtudeTurboWarpML',
      name: 'Etude-TurboWarp-ML',
      color1: '#4C97FF',
      color2: '#3d85c6',
      color3: '#2e5d8f',
      author: 'Asuka | Lin Xin',
      version: '0.0.1', 
      blocks: [
        // 核心模块
        {
          opcode: 'startModelDefinition',
          blockType: Scratch.BlockType.COMMAND,
          text: '开始模型定义，输入维度 [INPUT_DIM]',
          arguments: { INPUT_DIM: { type: Scratch.ArgumentType.NUMBER, defaultValue: 4 } },
          disableMonitor: true
        },
        {
          opcode: 'addLinearLayer',
          blockType: Scratch.BlockType.COMMAND,
          text: '添加线性层 输出维度 [OUTPUT_DIM] 激活函数 [ACTIVATION]',
          arguments: {
            OUTPUT_DIM: { type: Scratch.ArgumentType.NUMBER, defaultValue: 8 },
            ACTIVATION: { type: Scratch.ArgumentType.STRING, menu: 'ACTIVATION_MENU', defaultValue: 'relu' }
          },
          disableMonitor: true
        },
        {
          opcode: 'endModelDefinition',
          blockType: Scratch.BlockType.COMMAND,
          text: '结束模型定义',
          disableMonitor: true
        },
        {
          opcode: 'forward',
          blockType: Scratch.BlockType.REPORTER,
          text: '推理 输入向量 [INPUT]',
          arguments: { INPUT: { type: Scratch.ArgumentType.STRING, defaultValue: '[[1,2,3,4]]' } },
          disableMonitor: true
        },
        {
          opcode: 'getModelStructure',
          blockType: Scratch.BlockType.REPORTER,
          text: '获取模型结构JSON',
          disableMonitor: true
        },
        {
          opcode: 'clearModel',
          blockType: Scratch.BlockType.COMMAND,
          text: '清除模型',
          disableMonitor: true
        },
        {
          opcode: 'isModelDefined',
          blockType: Scratch.BlockType.BOOLEAN,
          text: '模型是否已定义',
          disableMonitor: true
        },
        
        { blockType: Scratch.BlockType.LABEL, text: '优化器模块' },
        // 优化器
        {
          opcode: 'stepSGD',
          blockType: Scratch.BlockType.COMMAND,
          text: 'SGD 预测值 [PRED] 真实值 [TARGET] 损失函数 [LOSS] 学习率 [LR]',
          arguments: {
            PRED: { type: Scratch.ArgumentType.STRING, defaultValue: '[[0.2, 0.8]]' },
            TARGET: { type: Scratch.ArgumentType.STRING, defaultValue: '[[0, 1]]' },
            LOSS: { type: Scratch.ArgumentType.STRING, menu: 'LOSS_MENU', defaultValue: 'mse' },
            LR: { type: Scratch.ArgumentType.NUMBER, defaultValue: 0.01 }
          },
          disableMonitor: true
        },
        
        { blockType: Scratch.BlockType.LABEL, text: '线性代数工具' },
        // 线性代数
        {
          opcode: 'matrixMultiplication',
          blockType: Scratch.BlockType.REPORTER,
          text: '矩阵 [A] × [B]',
          arguments: {
            A: { type: Scratch.ArgumentType.STRING, defaultValue: '[[1,2],[3,4]]' },
            B: { type: Scratch.ArgumentType.STRING, defaultValue: '[[5,6],[7,8]]' }
          },
          disableMonitor: true
        },
        {
          opcode: 'matrixAddition',
          blockType: Scratch.BlockType.REPORTER,
          text: '矩阵 [A] + [B]',
          arguments: {
            A: { type: Scratch.ArgumentType.STRING, defaultValue: '[[1,2],[3,4]]' },
            B: { type: Scratch.ArgumentType.STRING, defaultValue: '[[5,6],[7,8]]' }
          },
          disableMonitor: true
        }
      ],
      menus: {
        ACTIVATION_MENU: {
          acceptReporters: false,
          items: [
            { text: 'ReLU', value: 'relu' },
            { text: 'Tanh', value: 'tanh' },
            { text: 'Sigmoid', value: 'sigmoid' },
            { text: 'Softmax', value: 'softmax' },
            { text: '无', value: 'none' }
          ]
        },
        LOSS_MENU: {
          acceptReporters: false,
          items: [
            { text: '均方误差(MSE)', value: 'mse' },
            { text: '交叉熵', value: 'crossentropy' }
          ]
        }
      }
    };
  }

  // 代理方法
  startModelDefinition(args) { return this.core.startModelDefinition(args); }
  addLinearLayer(args) { return this.core.addLinearLayer(args); }
  endModelDefinition() { return this.core.endModelDefinition(); }
  forward(args) { return this.core.forward(args); }
  getModelStructure() { return this.core.getModelStructure(); }
  clearModel() { return this.core.clearModel(); }
  isModelDefined() { return this.core.isModelDefined(); }
  stepSGD(args) { return this.optimizer.stepSGD(args); }
  matrixMultiplication(args) { return this.linearAlgebra.matrix_multiplication(args); }
  matrixAddition(args) { return this.linearAlgebra.matrix_add(args); }
}

// 注册扩展
Scratch.extensions.register(new EtudeTurboWarpML());