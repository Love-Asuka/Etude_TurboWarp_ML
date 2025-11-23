const MLUtils = {
  transpose(matrix) {
    if (!matrix || !matrix[0]) return [];
    return matrix[0].map((_, col) => matrix.map(row => row[col]));
  },

  matMul(a, b) {
    if (!a || !b || a.length === 0 || b.length === 0) return [];
    if (a[0].length !== b.length) {
      console.error('[ML] 矩阵乘法维度不匹配');
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

  // Hadamard积
  hadamard(a, b) {
    if (!a || !b) return [];
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

  activationDerivative(inputData, activation_type, outputGrad) {
    if (!inputData || !outputGrad) return [];
    let inputGrad;
    switch(activation_type) {
      case 'relu':
        inputGrad = inputData.map(row => row.map(val => val > 0 ? 1 : 0));
        break;
      case 'tanh':
        inputGrad = inputData.map(row => row.map(val => 1 - val * val));
        break;
      case 'sigmoid':
        inputGrad = inputData.map(row => row.map(val => val * (1 - val)));
        break;
      case 'softmax':
        inputGrad = outputGrad;
        break;
      default:
        inputGrad = outputGrad;
    }
    return MLUtils.hadamard(inputGrad, outputGrad);
  }
};

class EtudeTurboWarpMLCore {
  constructor() {
    this.globalState = {
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

    this.globalState = {
      layers: [],
      currentInputDim: inputDim,
      isModelDefined: false,
      computationGraph: { forward: [], backward: [] },
      modelMeta: {
        name: '未命名模型',
        inputDim: inputDim,
        outputDim: null,
        totalLayers: 0,
        created: new Date().toISOString()
      },
      parameters: {},
      gradients: {},
      forwardData: {}
    };
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
    const graphId = `op_${String(layerIndex + 1).padStart(3, '0')}`;

    const layerConfig = {
      id: layerId,
      type: 'linear',
      input_dim: this.globalState.currentInputDim,
      output_dim: outputDim,
      activation: activation,
      graph_id: graphId
    };

    this.globalState.layers.push(layerConfig);

    const inputName = layerIndex === 0 ? 'input_vector' : `op_${String(layerIndex).padStart(3, '0')}_input`;
    const linearOutputName = `activation_${layerIndex + 1}`;

    this.globalState.computationGraph.forward.push({
      id: graphId,
      type: 'linear',
      inputs: [inputName],
      outputs: [linearOutputName],
      params: [`${layerId}.weight`, `${layerId}.bias`]
    });

    if (activation !== 'none') {
      const activationOutputName = layerIndex === this.globalState.layers.length - 1 ? 'final_output' : `op_${String(layerIndex + 2).padStart(3, '0')}_input`;
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

    console.log(`[core] 添加线性层: ${JSON.stringify(layerConfig)}`);
  }

  endModelDefinition() {
    if (this.globalState.layers.length === 0) {
      console.error('[core] 模型中至少需要一个层');
      return;
    }

    this.globalState.isModelDefined = true;
    const forwardGraph = this.globalState.computationGraph.forward;
    if (forwardGraph.length > 0) {
      const lastNode = forwardGraph[forwardGraph.length - 1];
      if (lastNode.type === 'activation') {
        lastNode.outputs = ['final_output'];
      } else if (lastNode.type === 'linear') {
        lastNode.outputs = ['output_logits'];
        forwardGraph.push({
          id: 'act_final',
          type: 'activation',
          activation_type: 'none',
          inputs: ['output_logits'],
          outputs: ['final_output']
        });
      }
    }

    this.globalState.parameters = {};
    this.globalState.forwardData = {};

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
      console.error('[core] 输入必须是二维数组');
      return '[]';
    }

    this.globalState.forwardData = {};
    this.globalState.forwardData['input_vector'] = input;
    let currentTensor = input;

    for (const node of this.globalState.computationGraph.forward) {
      if (node.type === 'linear') {
        currentTensor = this._linearForward(node, currentTensor);
      } else if (node.type === 'activation') {
        currentTensor = MLUtils.applyActivation(currentTensor, node.activation_type);
      }
      
      const outputName = node.outputs[0];
      this.globalState.forwardData[outputName] = currentTensor;
    }

    return JSON.stringify(currentTensor);
  }

  _linearForward(node, input) {
    const paramNames = node.params;
    const layerId = paramNames[0].split('.')[0];
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
    this.globalState = {
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
    console.log('[core] 模型已清除');
  }

  isModelDefined() {
    return this.globalState.isModelDefined;
  }
}
class EtudeTurboWarpMLAutograd {
  constructor(coreInstance) {
    this.core = coreInstance;
    this.gradBuffer = {};
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
    this.gradBuffer = {};
    
    const lastNode = tape[0];
    if (lastNode.type === 'activation') {
      this.gradBuffer[lastNode.inputs[0]] = outputGrad;
    } else {
      this.gradBuffer[lastNode.outputs[0]] = outputGrad;
    }

    for (const node of tape) {
      if (node.type === 'linear') {
        this._linearBackward(node);
      } else if (node.type === 'activation') {
        this._activationBackward(node);
      }
    }
    
    console.log('[autograd] 反向传播完成');
  }

  _linearBackward(node) {
    const inputName = node.inputs[0];
    const outputName = node.outputs[0];
    const outputGrad = this.gradBuffer[outputName];
    
    if (!outputGrad) return;

    const layerId = node.params[0].split('.')[0];
    const layer = this.core.globalState.layers.find(l => l.id === layerId);
    const inputData = this.core.globalState.forwardData?.[inputName];
    
    if (!layer || !inputData) return;

    const weightGrad = MLUtils.matMul(MLUtils.transpose(inputData), outputGrad);
    const biasGrad = MLUtils.sumRows(outputGrad);
    const inputGrad = MLUtils.matMul(outputGrad, MLUtils.transpose(this.core.globalState.parameters[layerId].weight));

    this.gradBuffer[inputName] = inputGrad;
    this.core.globalState.gradients[layerId] = { weight: weightGrad, bias: biasGrad };
  }

  _activationBackward(node) {
    const inputName = node.inputs[0];
    const outputName = node.outputs[0];
    const outputGrad = this.gradBuffer[outputName];
    
    if (!outputGrad) return;

    const inputData = this.core.globalState.forwardData?.[inputName];
    if (!inputData) return;

    this.gradBuffer[inputName] = MLUtils.activationDerivative(inputData, node.activation_type, outputGrad);
  }

  getParamGrad(args) {
    const grad = this.core.globalState.gradients[args.PARAM];
    return JSON.stringify(grad || { weight: [], bias: [] });
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
    this.learningRate = 0.01;
  }

  setLearningRate(args) {
    this.learningRate = parseFloat(args.LR) || 0.01;
    console.log(`[optimizer] 学习率设置为: ${this.learningRate}`);
  }

  stepSGD() {
    if (!this.core.globalState.isModelDefined) {
      console.error('[optimizer] 模型未定义');
      return;
    }

    this.core.globalState.layers.forEach(layer => {
      const layerId = layer.id;
      const layerGrad = this.core.globalState.gradients[layerId];
      const layerParams = this.core.globalState.parameters[layerId];
      
      if (!layerGrad || !layerParams) return;

      layerParams.weight = layerParams.weight.map((row, i) =>
        row.map((val, j) => val - this.learningRate * layerGrad.weight[i][j])
      );
      
      layerParams.bias = layerParams.bias.map((val, i) =>
        val - this.learningRate * layerGrad.bias[i]
      );
    });

    console.log('[optimizer] SGD步骤完成');
  }
}

// ==================== 线性代数模块 ====================
class EtudeTurboWarpMLLinearAlgebra {
  matrix_multiplication(args) {
    try {
      const a = JSON.parse(args.A);
      const b = JSON.parse(args.B);
      return JSON.stringify(MLUtils.matMul(a, b));
    } catch(e) {
      return '[]';
    }
  }

  matrix_add(args) {
    try {
      const a = JSON.parse(args.A);
      const b = JSON.parse(args.B);
      if (!a || !b || a.length !== b.length || a[0].length !== b[0].length) return '[]';
      const result = a.map((row, i) => row.map((val, j) => val + b[i][j]));
      return JSON.stringify(result);
    } catch(e) {
      return '[]';
    }
  }
}

// ==================== 主扩展类 ====================
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
      version: '1.0.4',
      blocks: [
        // 核心模块积木
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
        {
          blockType: Scratch.BlockType.LABEL,
          text: '自动微分模块'
        },
        // 自动微分积木
        {
          opcode: 'backward',
          blockType: Scratch.BlockType.COMMAND,
          text: '对输出梯度 [GRAD] 执行反向传播',
          arguments: { GRAD: { type: Scratch.ArgumentType.STRING, defaultValue: '[[1,0]]' } },
          disableMonitor: true
        },
        {
          opcode: 'getParamGrad',
          blockType: Scratch.BlockType.REPORTER,
          text: '获取 [PARAM] 的梯度',
          arguments: { PARAM: { type: Scratch.ArgumentType.STRING, menu: 'PARAM_MENU', defaultValue: 'layer_0_linear' } },
          disableMonitor: true
        },
        {
          opcode: 'zeroGrad',
          blockType: Scratch.BlockType.COMMAND,
          text: '清零所有梯度',
          disableMonitor: true
        },
        {
          blockType: Scratch.BlockType.LABEL,
          text: '优化器模块'
        },
        // 优化器积木
        {
          opcode: 'setLearningRate',
          blockType: Scratch.BlockType.COMMAND,
          text: '设置学习率 [LR]',
          arguments: { LR: { type: Scratch.ArgumentType.NUMBER, defaultValue: 0.01 } },
          disableMonitor: true
        },
        {
          opcode: 'stepSGD',
          blockType: Scratch.BlockType.COMMAND,
          text: '执行SGD优化步骤',
          disableMonitor: true
        },
        {
          blockType: Scratch.BlockType.LABEL,
          text: '线性代数工具'
        },
        // 线性代数积木
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
        PARAM_MENU: {
          acceptReporters: false,
          items: this._getParamMenu.bind(this)
        }
      }
    };
  }

  _getParamMenu() {
    try {
      const gradients = this.core.globalState.gradients;
      const keys = Object.keys(gradients);
      if (keys.length === 0) {
        return [{ text: '请先定义模型', value: 'none' }];
      }
      return keys.map(key => ({ text: key, value: key }));
    } catch(e) {
      return [{ text: '请先定义模型', value: 'none' }];
    }
  }

  // 核心模块方法代理
  startModelDefinition(args) { return this.core.startModelDefinition(args); }
  addLinearLayer(args) { return this.core.addLinearLayer(args); }
  endModelDefinition() { return this.core.endModelDefinition(); }
  forward(args) { return this.core.forward(args); }
  getModelStructure() { return this.core.getModelStructure(); }
  clearModel() { return this.core.clearModel(); }
  isModelDefined() { return this.core.isModelDefined(); }

  // 自动微分方法代理
  backward(args) { return this.autograd.backward(args); }
  getParamGrad(args) { return this.autograd.getParamGrad(args); }
  zeroGrad() { return this.autograd.zeroGrad(); }

  // 优化器方法代理
  setLearningRate(args) { return this.optimizer.setLearningRate(args); }
  stepSGD() { return this.optimizer.stepSGD(); }

  // 线性代数方法代理
  matrixMultiplication(args) { return this.linearAlgebra.matrix_multiplication(args); }
  matrixAddition(args) { return this.linearAlgebra.matrix_add(args); }
}

// 注册扩展
Scratch.extensions.register(new EtudeTurboWarpML());