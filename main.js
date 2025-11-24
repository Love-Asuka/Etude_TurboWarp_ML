const MLUtils = {
  Validation: {
    parseMatrix(str) {
      try {
        const parsed = JSON.parse(str);
        if (!Array.isArray(parsed) || !parsed[0] || !Array.isArray(parsed[0])) {
          return null;
        }
        return parsed;
      } catch (e) {
        return null;
      }
    },

    ensureModel(state) {
      return state && state.isModelDefined;
    },

    matchDims(actual, expected) {
      return actual === expected;
    },
    
    validatePositiveInt(val) {
      const num = parseInt(val);
      return (isNaN(num) || num <= 0) ? null : num;
    }
  },

  mapMatrices(a, b, operation) {
    if (!a || !b || a.length !== b.length || a[0].length !== b[0].length) return [];
    return a.map((row, i) => row.map((val, j) => operation(val, b[i][j])));
  },

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
            let sum = 0;
            for (let k = 0; k < n; k++) {
              const jacobian = j === k ? row[k] * (1 - row[k]) : -row[j] * row[k];
              sum += rowGrad[k] * jacobian;
            }
            result[j] = sum;
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
    if (!matrix || !matrix.length || !matrix[0]) return [];
    return matrix[0].map((_, col) => matrix.map(row => row[col]));
  },

  matMul(a, b) {
    if (!a.length || !b.length || !a[0] || !b[0] || a[0].length !== b.length) return [];
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

  computeLossAndGradient(pred, target, lossType) {
    if (!pred || !target || pred.length !== target.length) {
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
        let totalLoss = 0;
        const ceGrad = pred.map((row, i) => {
          let rowLoss = 0;
          const rowGrad = row.map((val, j) => {
            const t = target[i][j];
            const safeVal = val + epsilon;
            rowLoss -= t * Math.log(safeVal);
            return -t / safeVal; 
          });
          totalLoss += rowLoss;
          return rowGrad;
        });
        
        return { loss: totalLoss / batchSize, grad: ceGrad };
      }

      default:
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
        inputName: 'tensor_0',
        created: Date.now()
      },
      parameters: {},
      gradients: {},
      forwardData: {} 
    };
  }

  _generateGradientStructure(layer) {
    if (layer.type === 'linear') {
      return {
        weight: Array(layer.output_dim).fill().map(() => Array(layer.input_dim).fill(0)),
        bias: layer.use_bias ? Array(layer.output_dim).fill(0) : null
      };
    } else if (layer.type === 'layernorm') {
      return {
        weight: Array(layer.input_dim).fill(0), // Gamma
        bias: layer.use_bias ? Array(layer.input_dim).fill(0) : null // Beta
      };
    } else if (layer.type === 'rmsnorm') {
      return {
        weight: Array(layer.input_dim).fill(0), // Gamma
        bias: null
      };
    }
    return {};
  }

  addLinearLayer(args) {
    const outputDim = MLUtils.Validation.validatePositiveInt(args.OUTPUT_DIM);
    if (!outputDim) return;

    this._pendingLayers.push({
      type: 'linear',
      output_dim: outputDim,
      activation: args.ACTIVATION,
      use_bias: args.USE_BIAS === 'true'
    });
  }

  addLayerNorm(args) {
    this._pendingLayers.push({
      type: 'layernorm',
      use_bias: args.USE_BIAS === 'true'
    });
  }

  addRMSNorm(args) {
    this._pendingLayers.push({
      type: 'rmsnorm'
    });
  }

  endModelDefinition(args) {
    const inputDim = MLUtils.Validation.validatePositiveInt(args.INPUT_DIM);
    const initStrategy = args.INIT || 'he';

    if (!inputDim || this._pendingLayers.length === 0) return;

    this.globalState = this._createFreshState();
    this.globalState.modelMeta.inputDim = inputDim;
    
    let currentInputDim = inputDim;

    this._pendingLayers.forEach((layerConfig, index) => {
      const inputName = index === 0 ? 'tensor_0' : `tensor_${index}`;
      const outputName = `tensor_${index + 1}`; // Final output of this block
      const layerId = `layer_${index}_${layerConfig.type}`;

      // 统一的配置对象
      const fullLayerConfig = {
        id: layerId,
        type: layerConfig.type,
        input_dim: currentInputDim,
        output_dim: layerConfig.type === 'linear' ? layerConfig.output_dim : currentInputDim, // Norm layers keep dim
        activation: layerConfig.activation || 'none',
        use_bias: layerConfig.use_bias,
        input_name: inputName,
        output_name: outputName
      };

      this.globalState.layers.push(fullLayerConfig);

      // 构建参数和计算图
      if (layerConfig.type === 'linear') {
        const linearOutputName = `linear_${index + 1}`;
        const finalOutputName = layerConfig.activation !== 'none' ? outputName : linearOutputName;
        
        // Linear Op
        this.globalState.computationGraph.forward.push({
          id: `op_${index + 1}_lin`,
          type: 'linear',
          inputs: [inputName],
          outputs: [linearOutputName],
          layerId: layerId
        });

        // Activation Op (if any)
        if (layerConfig.activation !== 'none') {
            this.globalState.computationGraph.forward.push({
                id: `op_${index + 1}_act`,
                type: 'activation',
                activation_type: layerConfig.activation,
                inputs: [linearOutputName],
                outputs: [outputName]
            });
        } else {
            const lastNode = this.globalState.computationGraph.forward[this.globalState.computationGraph.forward.length-1];
            lastNode.outputs[0] = outputName;
        }
        const generator = MLUtils.Initializers[initStrategy] 
          ? MLUtils.Initializers[initStrategy](currentInputDim, layerConfig.output_dim)
          : MLUtils.Initializers.he(currentInputDim, layerConfig.output_dim);

        this.globalState.parameters[layerId] = {
          weight: Array(layerConfig.output_dim).fill().map(() => Array(currentInputDim).fill().map(generator)),
          bias: layerConfig.use_bias ? Array(layerConfig.output_dim).fill(0) : null
        };
        
        currentInputDim = layerConfig.output_dim;

      } else if (layerConfig.type === 'layernorm') {
        this.globalState.computationGraph.forward.push({
            id: `op_${index + 1}_ln`,
            type: 'layernorm',
            inputs: [inputName],
            outputs: [outputName],
            layerId: layerId
        });
        this.globalState.parameters[layerId] = {
            weight: Array(currentInputDim).fill(1), // Gamma
            bias: layerConfig.use_bias ? Array(currentInputDim).fill(0) : null // Beta
        };
        // Dims don't change

      } else if (layerConfig.type === 'rmsnorm') {
        this.globalState.computationGraph.forward.push({
            id: `op_${index + 1}_rms`,
            type: 'rmsnorm',
            inputs: [inputName],
            outputs: [outputName],
            layerId: layerId
        });
        this.globalState.parameters[layerId] = {
            weight: Array(currentInputDim).fill(1), // Gamma
            bias: null
        };
      }

      this.globalState.gradients[layerId] = this._generateGradientStructure(fullLayerConfig);
    });

    this.globalState.modelMeta.totalLayers = this.globalState.layers.length;
    this.globalState.modelMeta.outputDim = currentInputDim;
    this.globalState.isModelDefined = true;
    this._pendingLayers = [];
  }

  forward(args) {
    if (!MLUtils.Validation.ensureModel(this.globalState)) return '[]';
    
    const input = MLUtils.Validation.parseMatrix(args.INPUT);
    if (!input) return '[]';

    const expectedDim = this.globalState.modelMeta.inputDim;
    if (!MLUtils.Validation.matchDims(input[0].length, expectedDim)) return '[]';

    this.globalState.forwardData = {};
    
    let firstInputName = 'tensor_0';
    if (this.globalState.computationGraph.forward.length > 0) {
        firstInputName = this.globalState.computationGraph.forward[0].inputs[0];
    }
    this.globalState.forwardData[firstInputName] = { preActivation: null, postActivation: input };
    
    let currentTensor = input;

    for (const node of this.globalState.computationGraph.forward) {
      if (node.type === 'linear') {
        const out = this._linearForward(node, currentTensor);
        this.globalState.forwardData[node.outputs[0]] = { preActivation: currentTensor, postActivation: out };
        currentTensor = out;
      } else if (node.type === 'activation') {
        const activated = MLUtils.ActivationRegistry.apply(currentTensor, node.activation_type);
        this.globalState.forwardData[node.outputs[0]] = {
          preActivation: currentTensor,
          postActivation: activated
        };
        currentTensor = activated;
      } else if (node.type === 'layernorm') {
        const out = this._layerNormForward(node, currentTensor);
        currentTensor = out;
      } else if (node.type === 'rmsnorm') {
        const out = this._rmsNormForward(node, currentTensor);
        currentTensor = out;
      }
    }

    return JSON.stringify(currentTensor);
  }

  _linearForward(node, input) {
    const layerId = node.layerId;
    const layerParams = this.globalState.parameters[layerId];
    if (!layerParams) return input;

    return input.map(inputRow => {
        return layerParams.weight.map((weightRow, outIdx) => {
            let sum = 0;
            for (let k = 0; k < weightRow.length; k++) {
                sum += inputRow[k] * weightRow[k];
            }
            if (layerParams.bias) {
                sum += layerParams.bias[outIdx];
            }
            return sum;
        });
    });
  }

  _layerNormForward(node, input) {
    const layerId = node.layerId;
    const params = this.globalState.parameters[layerId];
    const gamma = params.weight;
    const beta = params.bias;   
    const epsilon = 1e-5;

    const cache = [];
    const output = input.map(row => {
        const n = row.length;
        const mean = row.reduce((a, b) => a + b, 0) / n;
        const variance = row.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
        const invStd = 1 / Math.sqrt(variance + epsilon);
        
        cache.push({ mean, invStd });

        return row.map((val, i) => {
            const normalized = (val - mean) * invStd;
            const scaled = normalized * gamma[i];
            return beta ? scaled + beta[i] : scaled;
        });
    });

    this.globalState.forwardData[node.outputs[0]] = {
        preActivation: input,
        postActivation: output,
        cache: cache 
    };

    return output;
  }

  _rmsNormForward(node, input) {
    const layerId = node.layerId;
    const params = this.globalState.parameters[layerId];
    const gamma = params.weight; // Vector
    const epsilon = 1e-5;

    const cache = [];
    const output = input.map(row => {
        const n = row.length;
        const meanSquare = row.reduce((a, b) => a + b * b, 0) / n;
        const invRms = 1 / Math.sqrt(meanSquare + epsilon);
        
        cache.push({ invRms });

        return row.map((val, i) => {
            return val * invRms * gamma[i];
        });
    });

    this.globalState.forwardData[node.outputs[0]] = {
        preActivation: input,
        postActivation: output,
        cache: cache
    };

    return output;
  }

  getModelStructure() {
    if (!this.globalState.isModelDefined) {
      return JSON.stringify({ error: '模型尚未定义' }, null, 2);
    }
    return JSON.stringify({
      format: 'etude-ml-model',
      version: '1.1',
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
      if (!modelData.format || !modelData.format.startsWith('etude-ml-model')) return;

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
          
          if (params?.weight) {
            newState.parameters[config.id] = { 
                weight: params.weight, 
                bias: params.bias !== undefined ? params.bias : null 
            };
          }
          newState.gradients[config.id] = this._generateGradientStructure(config);
        });
      }

      this.globalState = newState;
    } catch (e) {
      console.warn("Load model failed", e);
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
    if (!MLUtils.Validation.ensureModel(this.core.globalState)) return;
    
    const outputGrad = MLUtils.Validation.parseMatrix(args.GRAD);
    if (!outputGrad) return;

    const tape = [...this.core.globalState.computationGraph.forward].reverse();
    const gradBuffer = {};

    if (tape.length > 0) {
        gradBuffer[tape[0].outputs[0]] = outputGrad;
    }

    for (const node of tape) {
      const grad = gradBuffer[node.outputs[0]];

      if (!grad) continue;

      if (node.type === 'linear') {
        this._linearBackward(node, gradBuffer);
      } else if (node.type === 'activation') {
        this._activationBackward(node, gradBuffer);
      } else if (node.type === 'layernorm') {
        this._layerNormBackward(node, gradBuffer);
      } else if (node.type === 'rmsnorm') {
        this._rmsNormBackward(node, gradBuffer);
      }
    }
  }

  _linearBackward(node, gradBuffer) {
    const inputName = node.inputs[0];
    const outputName = node.outputs[0];
    const outputGrad = gradBuffer[outputName];
    
    const layerId = node.layerId;
    
    const fwdData = this.core.globalState.forwardData[outputName];
    if (!fwdData) return;
    
    const inputData = fwdData.preActivation; 
    const params = this.core.globalState.parameters[layerId];
    
    if (!inputData || !params || !params.weight) return;

    const weightGrad = MLUtils.matMul(MLUtils.transpose(outputGrad), inputData);

    let biasGrad = null;
    if (params.bias) {
        biasGrad = MLUtils.sumRows(outputGrad);
    }

    const inputGrad = MLUtils.matMul(outputGrad, params.weight);

    gradBuffer[inputName] = inputGrad;
    this.core.globalState.gradients[layerId] = { weight: weightGrad, bias: biasGrad };
  }

  _activationBackward(node, gradBuffer) {
    const outputName = node.outputs[0];
    const outputGrad = gradBuffer[outputName];
    
    const activationData = this.core.globalState.forwardData?.[outputName];
    if (!activationData) return;

    const inputGrad = MLUtils.ActivationRegistry.derivative(
      activationData.postActivation, 
      node.activation_type, 
      outputGrad
    );
    
    gradBuffer[node.inputs[0]] = inputGrad;
  }

  _layerNormBackward(node, gradBuffer) {
    const layerId = node.layerId;
    const outputName = node.outputs[0];
    const dy = gradBuffer[outputName]; // Shape: [Batch, Dim]
    
    const fwdData = this.core.globalState.forwardData[outputName];
    const x = fwdData.preActivation; // Input X
    const cache = fwdData.cache; // [{mean, invStd}, ...]
    const params = this.core.globalState.parameters[layerId];
    const gamma = params.weight; // Shape: [Dim]
    const hasBias = !!params.bias;
    
    const N = x.length;     // Batch size
    const D = x[0].length;  // Dimension

    const dGamma = new Array(D).fill(0);
    const dBeta = hasBias ? new Array(D).fill(0) : null;
    const dx = [];

    for (let i = 0; i < N; i++) {
        const row_dy = dy[i];
        const row_x = x[i];
        const { mean, invStd } = cache[i];
        

        const row_norm_x = row_x.map(val => (val - mean) * invStd);
        
        for (let j = 0; j < D; j++) {
            dGamma[j] += row_dy[j] * row_norm_x[j]; 
            if (hasBias) {
                dBeta[j] += row_dy[j];
            }                  
        }

        const dl_dxhat = row_dy.map((val, j) => val * gamma[j]);
        
        const sum_dl_dxhat_xhat = dl_dxhat.reduce((acc, val, j) => acc + val * row_norm_x[j], 0);
        const sum_dl_dxhat = dl_dxhat.reduce((acc, val) => acc + val, 0);

        const row_dx = dl_dxhat.map((val, j) => {
             return (1 / D) * invStd * (D * val - sum_dl_dxhat - row_norm_x[j] * sum_dl_dxhat_xhat);
        });
        dx.push(row_dx);
    }

    
    gradBuffer[node.inputs[0]] = dx;
    this.core.globalState.gradients[layerId] = { weight: dGamma, bias: dBeta };
  }

  _rmsNormBackward(node, gradBuffer) {
    const layerId = node.layerId;
    const outputName = node.outputs[0];
    const dy = gradBuffer[outputName];
    
    const fwdData = this.core.globalState.forwardData[outputName];
    const x = fwdData.preActivation;
    const cache = fwdData.cache; // [{invRms}, ...]
    const params = this.core.globalState.parameters[layerId];
    const gamma = params.weight;

    const N = x.length;
    const D = x[0].length;

    const dGamma = new Array(D).fill(0);
    const dx = [];

    for (let i = 0; i < N; i++) {
        const row_dy = dy[i];
        const row_x = x[i];
        const { invRms } = cache[i];


        for (let j = 0; j < D; j++) {
            dGamma[j] += row_dy[j] * (row_x[j] * invRms);
        }

        
        const g = row_dy.map((val, j) => val * gamma[j]);
        const sum_g_x = g.reduce((acc, val, j) => acc + val * row_x[j], 0);
        const factor = (invRms * invRms) / D * sum_g_x;
        
        const row_dx = g.map((val, j) => {
            return invRms * (val - row_x[j] * factor);
        });
        dx.push(row_dx);
    }

    gradBuffer[node.inputs[0]] = dx;
    this.core.globalState.gradients[layerId] = { weight: dGamma, bias: null };
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
    if (!MLUtils.Validation.ensureModel(this.core.globalState)) return;

    const pred = MLUtils.Validation.parseMatrix(args.PRED);
    const target = MLUtils.Validation.parseMatrix(args.TARGET);
    if (!pred || !target) return;

    const lossType = args.LOSS || 'mse';
    const learningRate = parseFloat(args.LR) || 0.01;

    this.autograd.zeroGrad();

    const { loss, grad } = MLUtils.computeLossAndGradient(pred, target, lossType);

    this.autograd.backward({ GRAD: JSON.stringify(grad) });

    this.core.globalState.layers.forEach(layer => {
      const layerId = layer.id;
      const layerGrad = this.core.globalState.gradients[layerId];
      const layerParams = this.core.globalState.parameters[layerId];
      
      if (!layerGrad || !layerParams) return;

      if (Array.isArray(layerParams.weight[0])) {
          for(let i=0; i<layerParams.weight.length; i++) {
              for(let j=0; j<layerParams.weight[i].length; j++) {
                  layerParams.weight[i][j] -= learningRate * layerGrad.weight[i][j];
              }
          }
      } else {
           for(let i=0; i<layerParams.weight.length; i++) {
              layerParams.weight[i] -= learningRate * layerGrad.weight[i];
           }
      }

      if (layerParams.bias && layerGrad.bias) {
        for(let i=0; i<layerParams.bias.length; i++) {
            layerParams.bias[i] -= learningRate * layerGrad.bias[i];
        }
      }
    });
  }
}

class EtudeTurboWarpMLLinearAlgebra {
  matrixMultiplication(args) {
    const a = MLUtils.Validation.parseMatrix(args.A);
    const b = MLUtils.Validation.parseMatrix(args.B);
    if (!a || !b) return '[]';
    return JSON.stringify(MLUtils.matMul(a, b));
  }
  
  matrixAddition(args) {
    const a = MLUtils.Validation.parseMatrix(args.A);
    const b = MLUtils.Validation.parseMatrix(args.B);
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
      version: '0.0.5',
      blocks: [
        { blockType: Scratch.BlockType.LABEL, text: '模型构建与管理' },
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
          opcode: 'addLinearLayer',
          blockType: Scratch.BlockType.COMMAND,
          text: '添加线性层 输出维度 [OUTPUT_DIM] 激活函数 [ACTIVATION] 使用偏置 [USE_BIAS]',
          arguments: {
            OUTPUT_DIM: { type: Scratch.ArgumentType.NUMBER, defaultValue: 4 },
            ACTIVATION: { type: Scratch.ArgumentType.STRING, menu: 'ACTIVATION_MENU', defaultValue: 'relu' },
            USE_BIAS: { type: Scratch.ArgumentType.STRING, menu: 'BOOL_MENU', defaultValue: 'true' }
          },
          disableMonitor: true
        },
        {
          opcode: 'addLayerNorm',
          blockType: Scratch.BlockType.COMMAND,
          text: '添加层归一化 (LayerNorm) 使用偏置 [USE_BIAS]',
          arguments: {
            USE_BIAS: { type: Scratch.ArgumentType.STRING, menu: 'BOOL_MENU', defaultValue: 'true' }
          },
          disableMonitor: true
        },
        {
          opcode: 'addRMSNorm',
          blockType: Scratch.BlockType.COMMAND,
          text: '添加RMS归一化 (RMSNorm)',
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
        },
        BOOL_MENU: {
          acceptReporters: false,
          items: [
            { text: '是', value: 'true' },
            { text: '否', value: 'false' }
          ]
        }
      }
    };
  }
}

Scratch.extensions.register(new EtudeTurboWarpML());