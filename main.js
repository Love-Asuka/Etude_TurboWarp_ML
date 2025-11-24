const MLUtils = {
  Validation: {
    parseMatrix(str, context = 'ML') {
      try {
        const parsed = JSON.parse(str);
        if (!Array.isArray(parsed) || !parsed[0] || !Array.isArray(parsed[0])) {
          throw new Error('无效的矩阵格式，应为二维数组');
        }
        return parsed;
      } catch (e) {
        console.error(`[${context}] 解析失败: ${e.message}`);
        return null;
      }
    },

    ensureModel(state, context) {
      if (!state.isModelDefined) {
        console.error(`[${context}] 模型未定义或未编译`);
        return false;
      }
      return true;
    },

    matchDims(actual, expected, context) {
      if (actual !== expected) {
        console.error(`[${context}] 维度不匹配: 期望 ${expected}, 得到 ${actual}`);
        return false;
      }
      return true;
    },
    
    validatePositiveInt(val, name, context = 'ML') {
      const num = parseInt(val);
      if (isNaN(num) || num <= 0) {
        console.error(`[${context}] ${name} 必须是正整数`);
        return null;
      }
      return num;
    }
  },

  mapMatrices(a, b, operation) {
    if (!a || !b || a.length !== b.length || a[0].length !== b[0].length) return [];
    return a.map((row, i) => row.map((val, j) => operation(val, b[i][j])));
  },

  // 激活函数注册表
  ActivationRegistry: {
    _activations: {
      relu: {
        forward: row => row.map(x => Math.max(0, x)),
        backward: (row, gradRow) => row.map((x, i) => (x > 0 ? gradRow[i] : 0))
      },
      tanh: {
        forward: row => row.map(x => Math.tanh(x)),
        backward: (row, gradRow) => row.map((x, i) => (1 - x * x) * gradRow[i])
      },
      sigmoid: {
        forward: row => row.map(x => 1 / (1 + Math.exp(-x))),
        backward: (row, gradRow) => row.map((x, i) => (x * (1 - x)) * gradRow[i])
      },
      softmax: {
        forward: row => {
          const maxVal = Math.max(...row);
          const exps = row.map(val => Math.exp(val - maxVal));
          const sumExps = exps.reduce((a, b) => a + b, 0);
          return exps.map(exp => exp / sumExps);
        },
        backward: (row, rowGrad) => {
          const n = row.length;
          const result = new Array(n).fill(0);
          for (let j = 0; j < n; j++) {
            for (let k = 0; k < n; k++) {

              const jacobian = j === k ? row[k] * (1 - row[k]) : -row[j] * row[k];
              result[j] += rowGrad[k] * jacobian;
            }
          }
          return result;
        }
      }
    },

    apply(matrix, type) {
      if (!matrix || !this._activations[type]) return matrix;
      return matrix.map(row => this._activations[type].forward(row));
    },

    derivative(activated, type, grad) {
      if (!activated || !grad || !this._activations[type]) return grad;
      return activated.map((row, i) => 
        this._activations[type].backward(row, grad[i])
      );
    }
  },

  Initializers: {
    he: (inDim, outDim) => {
      const scale = Math.sqrt(2.0 / inDim);
      return () => (Math.random() - 0.5) * 2 * scale;
    },
    xavier: (inDim, outDim) => {
      const limit = Math.sqrt(6 / (inDim + outDim));
      return () => (Math.random() - 0.5) * 2 * limit;
    },
    zeros: () => () => 0,
    ones: () => () => 1
  },

  transpose(matrix) {
    if (!matrix || !Array.isArray(matrix) || matrix.length === 0 || !Array.isArray(matrix[0])) return [];
    return matrix[0].map((_, col) => matrix.map(row => row[col]));
  },

  matMul(a, b) {
    if (!a || !b || a.length === 0 || b.length === 0 || !a[0] || !b[0]) return [];
    if (a[0].length !== b.length) {
      console.error(`[ML] 矩阵乘法维度不匹配: A[0].length=${a[0].length} ≠ B.length=${b.length}`);
      return [];
    }
    const bT = this.transpose(b); 
    return a.map(row => 
      bT.map(col => 
        row.reduce((sum, val, k) => sum + val * col[k], 0)
      )
    );
  },

  sumRows(matrix) {
    if (!matrix || !matrix[0]) return [];
    return matrix[0].map((_, col) => 
      matrix.reduce((sum, row) => sum + row[col], 0)
    );
  },

  matAdd(a, b) {
    return this.mapMatrices(a, b, (x, y) => x + y);
  },

  hadamard(a, b) {
    return this.mapMatrices(a, b, (x, y) => x * y);
  },

  computeLossAndGradient(pred, target, lossType) {
    if (!pred || !target || pred.length !== target.length) {
      console.error('[ML] 预测值与真实值维度不匹配');
      return { loss: 0, grad: [] };
    }

    const batchSize = pred.length;
    
    switch(lossType) {
      case 'mse': {
        let totalLoss = 0;
        const mseGrad = pred.map((row, i) => 
          row.map((val, j) => {
            const diff = val - target[i][j];
            totalLoss += diff * diff;
            return 2 * diff / batchSize;
          })
        );
        return { loss: totalLoss / batchSize, grad: mseGrad };
      }

      case 'crossentropy': {
        const epsilon = 1e-7;
        const ceLoss = -pred.reduce((sum, row, i) => 
          sum + row.reduce((rowSum, val, j) => rowSum + target[i][j] * Math.log(val + epsilon), 0), 0
        ) / batchSize;

        const ceGrad = pred.map((row, i) => 
          row.map((val, j) => val - target[i][j])
        );
        return { loss: ceLoss, grad: ceGrad };
      }

      default:
        console.error(`[ML] 不支持的损失函数: ${lossType}`);
        return { loss: 0, grad: [] };
    }
  }
};

class EtudeTurboWarpMLCore {
  constructor() {
    this._pendingLayers = [];
    this.globalState = this._createFreshState();
  }

  _createFreshState() {
    return {
      layers: [],
      isModelDefined: false,
      computationGraph: { forward: [], backward: [] },
      modelMeta: {
        name: 'Etude-Model',
        inputDim: null,
        outputDim: null,
        totalLayers: 0,
        created: Date.now()
      },
      parameters: {},
      gradients: {},
      forwardData: {} 
    };
  }

  _generateGradientStructure(layer) {
    return {
      weight: Array(layer.output_dim).fill().map(() => Array(layer.input_dim).fill(0)),
      bias: Array(layer.output_dim).fill(0)
    };
  }

  addLinearLayer(args) {
    const outputDim = MLUtils.Validation.validatePositiveInt(args.OUTPUT_DIM, '输出维度', 'core');
    if (!outputDim) return;

    this._pendingLayers.push({
      type: 'linear',
      output_dim: outputDim,
      activation: args.ACTIVATION
    });

    console.log(`[core] 层配置已添加: Out=${outputDim}, Act=${args.ACTIVATION}`);
  }

  endModelDefinition(args) {
    const inputDim = MLUtils.Validation.validatePositiveInt(args.INPUT_DIM, '输入维度', 'core');
    const initStrategy = args.INIT || 'he';

    if (!inputDim) return;
    if (this._pendingLayers.length === 0) {
      console.error('[core] 模型为空，请先添加层');
      return;
    }

    this.globalState = this._createFreshState();
    this.globalState.modelMeta.inputDim = inputDim;
    
    let currentInputDim = inputDim;

    this._pendingLayers.forEach((layerConfig, index) => {
      const layerId = `layer_${index}_linear`;
      const inputName = index === 0 ? 'tensor_0' : `tensor_${index}`;
      const linearOutputName = `linear_${index + 1}`;
      const activationOutputName = `tensor_${index + 1}`;

      const fullLayerConfig = {
        id: layerId,
        type: 'linear',
        input_dim: currentInputDim,
        output_dim: layerConfig.output_dim,
        activation: layerConfig.activation,
        input_name: inputName,
        output_name: activationOutputName
      };

      this.globalState.layers.push(fullLayerConfig);

      this.globalState.computationGraph.forward.push({
        id: `op_${index + 1}`,
        type: 'linear',
        inputs: [inputName],
        outputs: [linearOutputName],
        layerId: layerId
      });

      if (layerConfig.activation !== 'none') {
        this.globalState.computationGraph.forward.push({
          id: `act_${index + 1}`,
          type: 'activation',
          activation_type: layerConfig.activation,
          inputs: [linearOutputName],
          outputs: [activationOutputName]
        });
      }

      const generator = MLUtils.Initializers[initStrategy] 
        ? MLUtils.Initializers[initStrategy](currentInputDim, layerConfig.output_dim)
        : MLUtils.Initializers.he(currentInputDim, layerConfig.output_dim);

      this.globalState.parameters[layerId] = {
        weight: Array(layerConfig.output_dim).fill().map(() => Array(currentInputDim).fill().map(generator)),
        bias: Array(layerConfig.output_dim).fill(0)
      };

      this.globalState.gradients[layerId] = this._generateGradientStructure(fullLayerConfig);
      currentInputDim = layerConfig.output_dim;
    });

    this.globalState.modelMeta.totalLayers = this.globalState.layers.length;
    this.globalState.modelMeta.outputDim = currentInputDim;
    this.globalState.isModelDefined = true;
    this._pendingLayers = [];

    console.log(`[core] 模型构建完成。In: ${inputDim}, Out: ${currentInputDim}, Init: ${initStrategy}`);
  }

  forward(args) {
    if (!MLUtils.Validation.ensureModel(this.globalState, 'core')) return '[]';
    
    const input = MLUtils.Validation.parseMatrix(args.INPUT, 'core');
    if (!input) return '[]';

    const expectedDim = this.globalState.modelMeta.inputDim;
    if (!MLUtils.Validation.matchDims(input[0].length, expectedDim, 'core')) return '[]';

    this.globalState.forwardData = {};
    this.globalState.forwardData['tensor_0'] = { preActivation: null, postActivation: input };
    
    let currentTensor = input;

    for (const node of this.globalState.computationGraph.forward) {
      if (node.type === 'linear') {
        currentTensor = this._linearForward(node, currentTensor);
      } else if (node.type === 'activation') {
        const activated = MLUtils.ActivationRegistry.apply(currentTensor, node.activation_type);
        this.globalState.forwardData[node.outputs[0]] = {
          preActivation: currentTensor,
          postActivation: activated
        };
        currentTensor = activated;
      }
    }

    return JSON.stringify(currentTensor);
  }

  _linearForward(node, input) {
    const layerId = node.layerId;
    const layerParams = this.globalState.parameters[layerId];
    
    if (!layerParams) return input;

    const weightT = MLUtils.transpose(layerParams.weight);
    const output = MLUtils.matMul(input, weightT);
    
    return output.map(row => row.map((val, j) => val + layerParams.bias[j]));
  }

  getModelStructure() {
    if (!this.globalState.isModelDefined) {
      return JSON.stringify({ error: '模型尚未定义' }, null, 2);
    }
    return JSON.stringify({
      format: 'etude-ml-model',
      version: '1.0',
      meta: this.globalState.modelMeta,
      layers: this.globalState.layers.map(layer => ({
        config: layer,
        parameters: this.globalState.parameters[layer.id] || {}
      })),
      computation_graph: this.globalState.computationGraph
    });
  }

  loadModel(args) {
    try {
      const modelData = JSON.parse(args.JSON);
      if (modelData.format !== 'etude-ml-model') {
        console.error('[core] 无效的模型格式');
        return;
      }

      const newState = this._createFreshState();
      newState.modelMeta = modelData.meta;
      newState.computationGraph = modelData.computation_graph;
      newState.isModelDefined = true;
      this._pendingLayers = []; 

      if (Array.isArray(modelData.layers)) {
        modelData.layers.forEach(layerData => {
          const config = layerData.config;
          const params = layerData.parameters;
          newState.layers.push(config);
          
          if (params?.weight && params?.bias) {
            newState.parameters[config.id] = { weight: params.weight, bias: params.bias };
          }
          newState.gradients[config.id] = this._generateGradientStructure(config);
        });
      }

      this.globalState = newState;
      console.log(`[core] 模型已加载: ${newState.modelMeta.name}`);
    } catch (e) {
      console.error(`[core] 模型加载失败: ${e.message}`);
    }
  }

  clearModel() {
    this.globalState = this._createFreshState();
    this._pendingLayers = [];
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
    if (!MLUtils.Validation.ensureModel(this.core.globalState, 'autograd')) return;
    
    const outputGrad = MLUtils.Validation.parseMatrix(args.GRAD, 'autograd');
    if (!outputGrad) return;

    const tape = [...this.core.globalState.computationGraph.forward].reverse();
    const gradBuffer = {};

    const lastLayerOutDim = this.core.globalState.modelMeta.outputDim;
    if (outputGrad[0].length !== lastLayerOutDim) {
        console.error(`[autograd] 梯度维度错误。期望: ${lastLayerOutDim}, 实际: ${outputGrad[0].length}`);
        return;
    }

    gradBuffer[tape[0].outputs[0]] = outputGrad;

    for (const node of tape) {
      if (node.type === 'linear') {
        this._linearBackward(node, gradBuffer);
      } else if (node.type === 'activation') {
        this._activationBackward(node, gradBuffer);
      }
    }
  }

_linearBackward(node, gradBuffer) {
    const inputName = node.inputs[0];
    const outputName = node.outputs[0];
    const outputGrad = gradBuffer[outputName];
    if (!outputGrad) return;

    const layerId = node.layerId;
    const inputData = this.core.globalState.forwardData?.[inputName]?.postActivation;
    const weight = this.core.globalState.parameters[layerId]?.weight;
    
    if (!inputData || !weight) return;
    
    const weightGrad = MLUtils.matMul(MLUtils.transpose(outputGrad), inputData);

    const biasGrad = MLUtils.sumRows(outputGrad);

    const inputGrad = MLUtils.matMul(outputGrad, weight);

    gradBuffer[inputName] = inputGrad;
    this.core.globalState.gradients[layerId] = { weight: weightGrad, bias: biasGrad };
  }

  _activationBackward(node, gradBuffer) {
    const outputName = node.outputs[0];
    const outputGrad = gradBuffer[outputName];
    if (!outputGrad) return;

    const activationData = this.core.globalState.forwardData?.[outputName];
    if (!activationData) return;

    const inputGrad = MLUtils.ActivationRegistry.derivative(
      activationData.postActivation, 
      node.activation_type, 
      outputGrad
    );
    
    gradBuffer[node.inputs[0]] = inputGrad;
  }

  zeroGrad() {
    this.core.globalState.layers.forEach(layer => {
      this.core.globalState.gradients[layer.id] = this.core._generateGradientStructure(layer);
    });
  }
}

class EtudeTurboWarpMLOptimizer {
  constructor(coreInstance, autogradInstance) {
    this.core = coreInstance;
    this.autograd = autogradInstance;
  }

  stepSGD(args) {
    if (!MLUtils.Validation.ensureModel(this.core.globalState, 'optimizer')) return;

    const pred = MLUtils.Validation.parseMatrix(args.PRED, 'optimizer');
    const target = MLUtils.Validation.parseMatrix(args.TARGET, 'optimizer');
    if (!pred || !target) return;

    const lossType = args.LOSS || 'mse';
    const learningRate = parseFloat(args.LR) || 0.01;

    this.autograd.zeroGrad();

    const { loss, grad } = MLUtils.computeLossAndGradient(pred, target, lossType);
    console.log(`[optimizer] Loss (${lossType}): ${loss.toFixed(6)}`);

    this.autograd.backward({ GRAD: JSON.stringify(grad) });

    let updateCount = 0;
    this.core.globalState.layers.forEach(layer => {
      const layerId = layer.id;
      const layerGrad = this.core.globalState.gradients[layerId];
      const layerParams = this.core.globalState.parameters[layerId];
      
      if (!layerGrad || !layerParams) return;

      layerParams.weight = MLUtils.mapMatrices(
        layerParams.weight,
        layerGrad.weight,
        (w, g) => w - learningRate * g
      );
      
      layerParams.bias = layerParams.bias.map((b, i) => b - learningRate * layerGrad.bias[i]);
      
      updateCount++;
    });

    console.log(`[optimizer] SGD更新完成，层数: ${updateCount}`);
  }
}

class EtudeTurboWarpMLLinearAlgebra {
  matrixMultiplication(args) {
    const a = MLUtils.Validation.parseMatrix(args.A, 'matMul');
    const b = MLUtils.Validation.parseMatrix(args.B, 'matMul');
    if (!a || !b) return '[]';
    return JSON.stringify(MLUtils.matMul(a, b));
  }
  
  matrixAddition(args) {
    const a = MLUtils.Validation.parseMatrix(args.A, 'matAdd');
    const b = MLUtils.Validation.parseMatrix(args.B, 'matAdd');
    if (!a || !b) return '[]';
    return JSON.stringify(MLUtils.matAdd(a, b));
  }
}

class EtudeTurboWarpML {
  constructor() {
    this.core = new EtudeTurboWarpMLCore();
    this.autograd = new EtudeTurboWarpMLAutograd(this.core);
    this.optimizer = new EtudeTurboWarpMLOptimizer(this.core, this.autograd);
    this.linearAlgebra = new EtudeTurboWarpMLLinearAlgebra();
    
    this._setupAutoProxy();
  }

  _setupAutoProxy() {
    const modules = {
      core: this.core,
      optimizer: this.optimizer,
      linearAlgebra: this.linearAlgebra
    };

    Object.entries(modules).forEach(([_, module]) => {
      const methodNames = Object.getOwnPropertyNames(Object.getPrototypeOf(module))
        .filter(name => name !== 'constructor' && typeof module[name] === 'function' && !name.startsWith('_'));
      
      methodNames.forEach(methodName => {
        if (!this[methodName]) {
          this[methodName] = (...args) => module[methodName](...args);
        }
      });
    });
  }

  getInfo() {
    return {
      id: 'EtudeTurboWarpML',
      name: 'Etude-TurboWarp-ML',
      color1: '#4C97FF',
      color2: '#3d85c6',
      color3: '#2e5d8f',
      author: 'Asuka | Lin Xi',
      version: '0.0.3',
      blocks: [
        { blockType: Scratch.BlockType.LABEL, text: '模型构建与管理' },
        {
          opcode: 'addLinearLayer',
          blockType: Scratch.BlockType.COMMAND,
          text: '添加线性层 输出维度 [OUTPUT_DIM] 激活函数 [ACTIVATION]',
          arguments: {
            OUTPUT_DIM: { type: Scratch.ArgumentType.NUMBER, defaultValue: 4 },
            ACTIVATION: { type: Scratch.ArgumentType.STRING, menu: 'ACTIVATION_MENU', defaultValue: 'relu' }
          },
          disableMonitor: true
        },
        {
          opcode: 'endModelDefinition',
          blockType: Scratch.BlockType.COMMAND,
          text: '构建并初始化模型 输入维度 [INPUT_DIM] 策略 [INIT]',
          arguments: {
            INPUT_DIM: { type: Scratch.ArgumentType.NUMBER, defaultValue: 2 },
            INIT: { type: Scratch.ArgumentType.STRING, menu: 'INIT_MENU', defaultValue: 'he' }
          },
          disableMonitor: true
        },
        {
          opcode: 'loadModel',
          blockType: Scratch.BlockType.COMMAND,
          text: '加载模型 (JSON) [JSON]',
          arguments: { JSON: { type: Scratch.ArgumentType.STRING, defaultValue: '{}' } },
          disableMonitor: true
        },
        {
          opcode: 'getModelStructure',
          blockType: Scratch.BlockType.REPORTER,
          text: '导出模型 (JSON)',
          disableMonitor: true
        },
        {
          opcode: 'clearModel',
          blockType: Scratch.BlockType.COMMAND,
          text: '清除当前模型',
          disableMonitor: true
        },
        {
          opcode: 'isModelDefined',
          blockType: Scratch.BlockType.BOOLEAN,
          text: '模型已加载?',
          disableMonitor: true
        },
        
        { blockType: Scratch.BlockType.LABEL, text: '推理与训练' },
        {
          opcode: 'forward',
          blockType: Scratch.BlockType.REPORTER,
          text: '推理 输入向量 [INPUT]',
          arguments: { INPUT: { type: Scratch.ArgumentType.STRING, defaultValue: '[[1, 1]]' } },
          disableMonitor: true
        },
        {
          opcode: 'stepSGD',
          blockType: Scratch.BlockType.COMMAND,
          text: 'SGD优化 预测 [PRED] 目标 [TARGET] 损失 [LOSS] LR [LR]',
          arguments: {
            PRED: { type: Scratch.ArgumentType.STRING, defaultValue: '[[0]]' },
            TARGET: { type: Scratch.ArgumentType.STRING, defaultValue: '[[1]]' },
            LOSS: { type: Scratch.ArgumentType.STRING, menu: 'LOSS_MENU', defaultValue: 'mse' },
            LR: { type: Scratch.ArgumentType.NUMBER, defaultValue: 0.01 }
          },
          disableMonitor: true
        },
        
        { blockType: Scratch.BlockType.LABEL, text: '线性代数' },
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
        },
        INIT_MENU: {
          acceptReporters: false,
          items: [
            { text: 'He (ReLU推荐)', value: 'he' },
            { text: 'Xavier (Sigmoid/Tanh)', value: 'xavier' },
            { text: '全零', value: 'zeros' },
            { text: '全一', value: 'ones' }
          ]
        }
      }
    };
  }
}

Scratch.extensions.register(new EtudeTurboWarpML());