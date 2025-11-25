const MLUtils = {
  Validation: {
    parseMatrix(str) {
      try {
        const parsed = JSON.parse(str);
        if (Array.isArray(parsed) && Array.isArray(parsed[0])) return parsed;
        return null;
      } catch (e) {
        return null;
      }
    },

    ensureModel(state) {
      return state && state.isModelDefined;
    },

    validatePositiveInt(val) {
      const num = parseInt(val);
      return (isNaN(num) || num <= 0) ? null : num;
    }
  },


  zeros(shape) {
    if (shape.length === 1) return new Array(shape[0]).fill(0);
    return new Array(shape[0]).fill(0).map(() => new Array(shape[1]).fill(0));
  },

  tensorApply(target, source, fn, ...auxSources) {
    if (!target || !source) return;
    const is2D = Array.isArray(target[0]);
    
    if (is2D) {
      for (let i = 0; i < target.length; i++) {
        for (let j = 0; j < target[i].length; j++) {
          const auxArgs = auxSources.map(aux => aux ? aux[i][j] : undefined);
          target[i][j] = fn(target[i][j], source[i][j], ...auxArgs, i, j);
        }
      }
    } else {
      for (let i = 0; i < target.length; i++) {
        const auxArgs = auxSources.map(aux => aux ? aux[i] : undefined);
        target[i] = fn(target[i], source[i], ...auxArgs, i);
      }
    }
  },

  mapMatrices(a, b, operation) {
    if (!a || !b || a.length !== b.length || a[0].length !== b[0].length) return [];
    return a.map((row, i) => row.map((val, j) => operation(val, b[i][j])));
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


  ActivationRegistry: {
    _activations: {
      relu: {
        forward: x => Math.max(0, x),
        backward: (x, g) => (x > 0 ? g : 0)
      },
      tanh: {
        forward: x => Math.tanh(x),
        backward: (x, g) => (1 - x * x) * g
      },
      sigmoid: {
        forward: x => 1 / (1 + Math.exp(-x)),
        backward: (x, g) => (x * (1 - x)) * g
      },
      softmax: null 
    },

    apply(matrix, type) {
      if (!matrix) return matrix;
      if (type === 'softmax') {
        return matrix.map(row => {
          const maxVal = Math.max(...row);
          const exps = row.map(val => Math.exp(val - maxVal));
          const sumExps = exps.reduce((a, b) => a + b, 0);
          return exps.map(exp => exp / sumExps);
        });
      }

      const act = this._activations[type];
      if (!act) return matrix;
      return matrix.map(row => row.map(act.forward));
    },

    derivative(activated, type, grad) {
      if (!activated || !grad) return grad;

      if (type === 'softmax') {
        return activated.map((row, i) => {
          const rowGrad = grad[i];
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
        });
      }

      const act = this._activations[type];
      if (!act) return grad;
      return activated.map((row, i) => row.map((val, j) => act.backward(val, grad[i][j])));
    }
  },

  Initializers: {
    generate(strategy, inDim, outDim) {
      let generator;
      if (strategy === 'he') {
        const scale = Math.sqrt(2.0 / inDim);
        generator = () => (Math.random() - 0.5) * 2 * scale;
      } else if (strategy === 'xavier') {
        const limit = Math.sqrt(6 / (inDim + outDim));
        generator = () => (Math.random() - 0.5) * 2 * limit;
      } else if (strategy === 'ones') {
        generator = () => 1;
      } else { 
        generator = () => 0;
      }
      return Array(outDim).fill().map(() => Array(inDim).fill().map(generator));
    }
  },

  computeLossAndGradient(pred, target, lossType) {
    if (!pred || !target || pred.length !== target.length) {
      return { loss: 0, grad: [] };
    }

    const batchSize = pred.length;
    let totalLoss = 0;
    let grad = [];

    if (lossType === 'crossentropy') {
      const epsilon = 1e-7;
      grad = pred.map((row, i) => {
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
    } else { 
      grad = pred.map((row, i) => 
        row.map((val, j) => {
          const diff = val - target[i][j];
          totalLoss += diff * diff;
          return 2 * diff / batchSize;
        })
      );
    }
    
    return { loss: totalLoss / batchSize, grad };
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
      forwardData: {},
      optimizerState: { step: 0, m: {}, v: {} }
    };
  }

  _generateZeroGrads(layer) {
    const grads = {};
    if (layer.type === 'linear') {
      grads.weight = MLUtils.zeros([layer.output_dim, layer.input_dim]);
      grads.bias = layer.use_bias ? MLUtils.zeros([layer.output_dim]) : null;
    } else if (layer.type === 'layernorm' || layer.type === 'rmsnorm') {
      grads.weight = MLUtils.zeros([layer.input_dim]);
      grads.bias = (layer.use_bias) ? MLUtils.zeros([layer.input_dim]) : null;
    }
    return grads;
  }

  _addPendingLayer(config) {
    this._pendingLayers.push(config);
  }

  addLinearLayer(args) {
    const outputDim = MLUtils.Validation.validatePositiveInt(args.维度);
    if (!outputDim) return;
    this._addPendingLayer({
      type: 'linear',
      output_dim: outputDim,
      activation: args.ACTIVATION,
      use_bias: args.USE_BIAS === 'true'
    });
  }

  addLayerNorm(args) {
    this._addPendingLayer({
      type: 'layernorm',
      use_bias: args.USE_BIAS === 'true'
    });
  }

  addRMSNorm(args) {
    this._addPendingLayer({ type: 'rmsnorm' });
  }

  endModelDefinition(args) {
    if (this._pendingLayers.length === 0) return;


    const firstLinearIndex = this._pendingLayers.findIndex(l => l.type === 'linear');
    if (firstLinearIndex === -1) return;

    const inputDim = this._pendingLayers[firstLinearIndex].output_dim; 


    this.globalState = this._createFreshState();
    this.globalState.modelMeta.inputDim = inputDim;
    let currentInputDim = inputDim;

    this._pendingLayers.forEach((layerConfig, index) => {
      const inputName = index === 0 ? 'tensor_0' : `tensor_${index}`;
      const outputName = `tensor_${index + 1}`;
      const layerId = `layer_${index}_${layerConfig.type}`;
      
      const actualOutputDim = (layerConfig.type === 'linear' && index !== firstLinearIndex) 
        ? layerConfig.output_dim 
        : currentInputDim;

      const fullConfig = {
        id: layerId,
        type: layerConfig.type,
        input_dim: currentInputDim,
        output_dim: actualOutputDim,
        activation: layerConfig.activation || 'none',
        use_bias: layerConfig.use_bias,
        input_name: inputName,
        output_name: outputName
      };

      this.globalState.layers.push(fullConfig);
      
      if (layerConfig.type === 'linear') {
        const linearOutputName = layerConfig.activation !== 'none' ? `linear_${index + 1}` : outputName;
        
        this.globalState.computationGraph.forward.push({
          id: `op_${index}_lin`, type: 'linear', layerId,
          inputs: [inputName], outputs: [linearOutputName]
        });

        if (layerConfig.activation !== 'none') {
          this.globalState.computationGraph.forward.push({
            id: `op_${index}_act`, type: 'activation', activation_type: layerConfig.activation,
            inputs: [linearOutputName], outputs: [outputName]
          });
        }

        const weights = MLUtils.Initializers.generate(args.INIT || 'he', currentInputDim, actualOutputDim);
        this.globalState.parameters[layerId] = {
          weight: weights,
          bias: layerConfig.use_bias ? new Array(actualOutputDim).fill(0) : null
        };

        if (index !== firstLinearIndex) currentInputDim = actualOutputDim;

      } else {
        const isRMS = layerConfig.type === 'rmsnorm';
        this.globalState.computationGraph.forward.push({
          id: `op_${index}_norm`, type: layerConfig.type, layerId,
          inputs: [inputName], outputs: [outputName]
        });

        this.globalState.parameters[layerId] = {
          weight: new Array(currentInputDim).fill(1),
          bias: (layerConfig.use_bias && !isRMS) ? new Array(currentInputDim).fill(0) : null
        };
      }

      this.globalState.gradients[layerId] = this._generateZeroGrads(fullConfig);
    });

    this.globalState.modelMeta.totalLayers = this.globalState.layers.length;
    this.globalState.modelMeta.outputDim = currentInputDim;
    this.globalState.isModelDefined = true;
    this._pendingLayers = [];
  }

  forward(args) {
    if (!MLUtils.Validation.ensureModel(this.globalState)) return '[]';
    const input = MLUtils.Validation.parseMatrix(args.INPUT);
    if (!input || input[0].length !== this.globalState.modelMeta.inputDim) return '[]';

    this.globalState.forwardData = {};
    const firstInput = this.globalState.computationGraph.forward[0]?.inputs[0] || 'tensor_0';
    
    this.globalState.forwardData[firstInput] = { preActivation: null, postActivation: input };
    let currentVal = input;

    for (const node of this.globalState.computationGraph.forward) {
      const outputName = node.outputs[0];
      let outputVal;

      if (node.type === 'linear') {
        outputVal = this._linearForward(node, currentVal);
        this.globalState.forwardData[outputName] = { preActivation: currentVal, postActivation: outputVal };
      } else if (node.type === 'activation') {
        outputVal = MLUtils.ActivationRegistry.apply(currentVal, node.activation_type);
        this.globalState.forwardData[outputName] = { preActivation: currentVal, postActivation: outputVal };
      } else if (node.type === 'layernorm' || node.type === 'rmsnorm') {
        outputVal = this._normForward(node, currentVal); 
      }
      currentVal = outputVal;
    }

    return JSON.stringify(currentVal);
  }

  _linearForward(node, input) {
    const params = this.globalState.parameters[node.layerId];
    return input.map(row => 
      params.weight.map((wRow, idx) => {
        const dot = row.reduce((sum, val, k) => sum + val * wRow[k], 0);
        return params.bias ? dot + params.bias[idx] : dot;
      })
    );
  }

  _normForward(node, input) {
    const params = this.globalState.parameters[node.layerId];
    const isRMS = node.type === 'rmsnorm';
    const epsilon = 1e-5;
    const cache = [];

    const output = input.map(row => {
      const n = row.length;
      let mean = 0, variance = 0, invFactor = 0;

      if (isRMS) {
        const meanSquare = row.reduce((a, b) => a + b * b, 0) / n;
        invFactor = 1 / Math.sqrt(meanSquare + epsilon);
        cache.push({ invRms: invFactor });
      } else {
        mean = row.reduce((a, b) => a + b, 0) / n;
        variance = row.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
        invFactor = 1 / Math.sqrt(variance + epsilon);
        cache.push({ mean, invStd: invFactor });
      }

      return row.map((val, i) => {
        const normalized = isRMS ? val * invFactor : (val - mean) * invFactor;
        const scaled = normalized * params.weight[i];
        return (params.bias) ? scaled + params.bias[i] : scaled;
      });
    });

    this.globalState.forwardData[node.outputs[0]] = {
      preActivation: input, postActivation: output, cache
    };
    return output;
  }

  getModelStructure() {
    if (!this.globalState.isModelDefined) return JSON.stringify({ error: '模型未定义' }, null, 2);
    return JSON.stringify({
      format: 'etude-ml-model',
      version: '1.2',
      meta: this.globalState.modelMeta,
      layers: this.globalState.layers.map(layer => ({
        config: layer,
        parameters: this.globalState.parameters[layer.id]
      })),
      computation_graph: this.globalState.computationGraph
    });
  }

  loadModel(args) {
    try {
      const data = JSON.parse(args.JSON);
      if (!data.format?.startsWith('etude-ml-model')) return;

      const newState = this._createFreshState();
      newState.modelMeta = data.meta;
      newState.computationGraph = data.computation_graph;
      newState.isModelDefined = true;

      (data.layers || []).forEach(({ config, parameters }) => {
        newState.layers.push(config);
        if (parameters) newState.parameters[config.id] = parameters;
        newState.gradients[config.id] = this._generateZeroGrads(config);
      });

      this.globalState = newState;
    } catch (e) {
      console.warn("Load failed", e);
    }
  }

  clearModel() { this.globalState = this._createFreshState(); this._pendingLayers = []; }
  isModelDefined() { return this.globalState.isModelDefined; }
}

class EtudeTurboWarpMLAutograd {
  constructor(core) { this.core = core; }

  zeroGrad() {
    this.core.globalState.layers.forEach(layer => {
      this.core.globalState.gradients[layer.id] = this.core._generateZeroGrads(layer);
    });
  }

  backward(args) {
    if (!MLUtils.Validation.ensureModel(this.core.globalState)) return;
    const outputGrad = MLUtils.Validation.parseMatrix(args.GRAD);
    if (!outputGrad) return;

    const graph = this.core.globalState.computationGraph;
    const gradMap = { [graph.forward[graph.forward.length - 1].outputs[0]]: outputGrad };

    for (let i = graph.forward.length - 1; i >= 0; i--) {
      const node = graph.forward[i];
      const grad = gradMap[node.outputs[0]];
      
      if (!grad) continue;
      
      if (node.type === 'linear') this._linearBackward(node, grad, gradMap);
      else if (node.type === 'activation') this._activationBackward(node, grad, gradMap);
      else if (node.type === 'layernorm') this._normBackward(node, grad, gradMap, false);
      else if (node.type === 'rmsnorm') this._normBackward(node, grad, gradMap, true);
    }
  }

  _linearBackward(node, outputGrad, gradMap) {
    const fwd = this.core.globalState.forwardData[node.outputs[0]];
    const params = this.core.globalState.parameters[node.layerId];
    if (!fwd || !params) return;

    const input = fwd.preActivation;

    const dW = MLUtils.matMul(MLUtils.transpose(outputGrad), input);

    const dX = MLUtils.matMul(outputGrad, params.weight);
    
    gradMap[node.inputs[0]] = dX;
    
    this.core.globalState.gradients[node.layerId] = {
      weight: dW,
      bias: params.bias ? MLUtils.sumRows(outputGrad) : null
    };
  }

  _activationBackward(node, grad, gradMap) {
    const fwd = this.core.globalState.forwardData[node.outputs[0]];
    const dX = MLUtils.ActivationRegistry.derivative(fwd.postActivation, node.activation_type, grad);
    gradMap[node.inputs[0]] = dX;
  }

  _normBackward(node, dy, gradMap, isRMS) {
    const fwd = this.core.globalState.forwardData[node.outputs[0]];
    const params = this.core.globalState.parameters[node.layerId];
    if (!fwd || !params) return;

    const x = fwd.preActivation;
    const cache = fwd.cache;
    const gamma = params.weight;
    const N = x.length;
    const D = x[0].length;

    const dGamma = new Array(D).fill(0);
    const dBeta = params.bias ? new Array(D).fill(0) : null;
    const dx = [];

    for (let i = 0; i < N; i++) {
      const r_dy = dy[i];
      const r_x = x[i];
      const c = cache[i];
      
      const invFactor = isRMS ? c.invRms : c.invStd;
      const r_norm = r_x.map((val, k) => isRMS 
        ? val * invFactor 
        : (val - c.mean) * invFactor
      );

      for (let j = 0; j < D; j++) {
        dGamma[j] += r_dy[j] * r_norm[j];
        if (dBeta) dBeta[j] += r_dy[j];
      }

      const dl_dxhat = r_dy.map((val, j) => val * gamma[j]);
      
      let row_dx;
      if (isRMS) {
        const sum_g_x = dl_dxhat.reduce((acc, val, j) => acc + val * r_x[j], 0);
        const factor = (invFactor * invFactor) / D * sum_g_x;
        row_dx = dl_dxhat.map((val, j) => invFactor * (val - r_x[j] * factor));
      } else {
        const sum_dl_dxhat_xhat = dl_dxhat.reduce((acc, val, j) => acc + val * r_norm[j], 0);
        const sum_dl_dxhat = dl_dxhat.reduce((acc, val) => acc + val, 0);
        row_dx = dl_dxhat.map((val, j) => 
          (invFactor / D) * (D * val - sum_dl_dxhat - r_norm[j] * sum_dl_dxhat_xhat)
        );
      }
      dx.push(row_dx);
    }

    gradMap[node.inputs[0]] = dx;
    this.core.globalState.gradients[node.layerId] = { weight: dGamma, bias: dBeta };
  }
}

class EtudeTurboWarpMLOptimizer {
  constructor(core, autograd) {
    this.core = core;
    this.autograd = autograd;
  }

  _performOptimization(args, updateFunction) {
    if (!MLUtils.Validation.ensureModel(this.core.globalState)) return;
    const pred = MLUtils.Validation.parseMatrix(args.PRED);
    const target = MLUtils.Validation.parseMatrix(args.TARGET);
    if (!pred || !target) return;


    this.autograd.zeroGrad();

    const { grad } = MLUtils.computeLossAndGradient(pred, target, args.LOSS || 'mse');

    this.autograd.backward({ GRAD: JSON.stringify(grad) });

    const optState = this.core.globalState.optimizerState;
    optState.step++;

    this.core.globalState.layers.forEach(layer => {
      const pid = layer.id;
      const params = this.core.globalState.parameters[pid];
      const grads = this.core.globalState.gradients[pid];
      if (!params || !grads) return;

      if (!optState.m[pid]) optState.m[pid] = { weight: null, bias: null };
      if (!optState.v[pid]) optState.v[pid] = { weight: null, bias: null };

      this._updateParamTensor(params.weight, grads.weight, optState.m[pid], optState.v[pid], 'weight', updateFunction);
      
      if (params.bias) {
        this._updateParamTensor(params.bias, grads.bias, optState.m[pid], optState.v[pid], 'bias', updateFunction);
      }
    });
  }

  _updateParamTensor(pTensor, gTensor, mContainer, vContainer, key, updateFn) {
    if (!mContainer[key]) mContainer[key] = MLUtils.zeros(Array.isArray(pTensor[0]) ? [pTensor.length, pTensor[0].length] : [pTensor.length]);
    if (!vContainer[key]) vContainer[key] = MLUtils.zeros(Array.isArray(pTensor[0]) ? [pTensor.length, pTensor[0].length] : [pTensor.length]);

    MLUtils.tensorApply(
      pTensor, gTensor, 
      updateFn, 
      mContainer[key], vContainer[key]
    );
  }

  stepSGD(args) {
    const lr = parseFloat(args.LR) || 0.01;
    this._performOptimization(args, (param, grad) => {
      return param - lr * grad;
    });
  }

  _updateAdamW(pTensor, gTensor, mTensor, vTensor, config, t) {
      const c1 = 1 - Math.pow(config.beta1, t);
      const c2 = 1 - Math.pow(config.beta2, t);
      const is2D = Array.isArray(pTensor[0]);

      const update = (w, g, m, v) => {
          const nm = config.beta1 * m + (1 - config.beta1) * g;       
          const nv = config.beta2 * v + (1 - config.beta2) * (g * g); 
          
          const mHat = nm / c1; 
          const vHat = nv / c2;
          
          const nw = w - config.lr * ( (mHat / (Math.sqrt(vHat) + config.eps)) + config.decay * w );
          
          return { w: nw, m: nm, v: nv };
      };

      if (is2D) {
          for(let i=0; i<pTensor.length; i++) {
              for(let j=0; j<pTensor[i].length; j++) {
                  const res = update(pTensor[i][j], gTensor[i][j], mTensor[i][j], vTensor[i][j]);
                  pTensor[i][j] = res.w;
                  mTensor[i][j] = res.m; 
                  vTensor[i][j] = res.v;
              }
          }
      } else {
          for(let i=0; i<pTensor.length; i++) {
              const res = update(pTensor[i], gTensor[i], mTensor[i], vTensor[i]);
              pTensor[i] = res.w;
              mTensor[i] = res.m;
              vTensor[i] = res.v;
          }
      }
  }

  stepAdamW(args) {
    const config = {
        lr: parseFloat(args.LR) || 0.001,
        beta1: parseFloat(args.BETA1) || 0.9,
        beta2: parseFloat(args.BETA2) || 0.999,
        eps: parseFloat(args.EPS) || 1e-8,
        decay: parseFloat(args.DECAY) || 0.01
    };

    if (!MLUtils.Validation.ensureModel(this.core.globalState)) return;
    const pred = MLUtils.Validation.parseMatrix(args.PRED);
    const target = MLUtils.Validation.parseMatrix(args.TARGET);
    if (!pred || !target) return;

    this.autograd.zeroGrad();
    const { grad } = MLUtils.computeLossAndGradient(pred, target, args.LOSS || 'mse');
    this.autograd.backward({ GRAD: JSON.stringify(grad) });

    const optState = this.core.globalState.optimizerState;
    optState.step++;

    this.core.globalState.layers.forEach(layer => {
        const pid = layer.id;
        const params = this.core.globalState.parameters[pid];
        const grads = this.core.globalState.gradients[pid];
        if (!params || !grads) return;

        if (!optState.m[pid]) optState.m[pid] = { weight: null, bias: null };
        if (!optState.v[pid]) optState.v[pid] = { weight: null, bias: null };
        
        const initAux = (p, key) => {
             if (!optState.m[pid][key]) {
                 const shape = Array.isArray(p[0]) ? [p.length, p[0].length] : [p.length];
                 optState.m[pid][key] = MLUtils.zeros(shape);
                 optState.v[pid][key] = MLUtils.zeros(shape);
             }
        };

        initAux(params.weight, 'weight');
        this._updateAdamW(params.weight, grads.weight, optState.m[pid].weight, optState.v[pid].weight, config, optState.step);

        if (params.bias) {
            initAux(params.bias, 'bias');
            this._updateAdamW(params.bias, grads.bias, optState.m[pid].bias, optState.v[pid].bias, config, optState.step);
        }
    });
  }
}

class EtudeTurboWarpMLLinearAlgebra {
  matrixMultiplication(args) {
    const a = MLUtils.Validation.parseMatrix(args.A);
    const b = MLUtils.Validation.parseMatrix(args.B);
    return JSON.stringify(a && b ? MLUtils.matMul(a, b) : []);
  }
  
  matrixAddition(args) {
    const a = MLUtils.Validation.parseMatrix(args.A);
    const b = MLUtils.Validation.parseMatrix(args.B);
    return JSON.stringify(a && b ? MLUtils.matAdd(a, b) : []);
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
    const modules = { core: this.core, optimizer: this.optimizer, linearAlgebra: this.linearAlgebra };
    Object.entries(modules).forEach(([_, module]) => {
      Object.getOwnPropertyNames(Object.getPrototypeOf(module))
        .filter(n => n !== 'constructor' && typeof module[n] === 'function' && !n.startsWith('_'))
        .forEach(n => { if (!this[n]) this[n] = (...args) => module[n](...args); });
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
      version: '0.0.7',
      blocks: [
        { blockType: Scratch.BlockType.LABEL, text: '模型构建与管理' },
        {
          opcode: 'endModelDefinition',
          blockType: Scratch.BlockType.COMMAND,
          text: '构建并初始化模型 策略 [INIT]',
          arguments: { INIT: { type: Scratch.ArgumentType.STRING, menu: 'INIT_MENU', defaultValue: 'he' } },
          disableMonitor: true
        },
        {
          opcode: 'addLinearLayer',
          blockType: Scratch.BlockType.COMMAND,
          text: '添加线性层 维度 [维度] 激活函数 [ACTIVATION] 使用偏置 [USE_BIAS]',
          arguments: {
            维度: { type: Scratch.ArgumentType.NUMBER, defaultValue: 4 },
            ACTIVATION: { type: Scratch.ArgumentType.STRING, menu: 'ACTIVATION_MENU', defaultValue: 'relu' },
            USE_BIAS: { type: Scratch.ArgumentType.STRING, menu: 'BOOL_MENU', defaultValue: 'true' }
          },
          disableMonitor: true
        },
        {
          opcode: 'addLayerNorm',
          blockType: Scratch.BlockType.COMMAND,
          text: '添加层归一化 (LayerNorm) 使用偏置 [USE_BIAS]',
          arguments: { USE_BIAS: { type: Scratch.ArgumentType.STRING, menu: 'BOOL_MENU', defaultValue: 'true' } },
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
        {
          opcode: 'stepAdamW',
          blockType: Scratch.BlockType.COMMAND,
          text: 'AdamW优化 预测 [PRED] 目标 [TARGET] 损失 [LOSS] LR [LR] Decay [DECAY]',
          arguments: {
            PRED: { type: Scratch.ArgumentType.STRING, defaultValue: '[[0]]' },
            TARGET: { type: Scratch.ArgumentType.STRING, defaultValue: '[[1]]' },
            LOSS: { type: Scratch.ArgumentType.STRING, menu: 'LOSS_MENU', defaultValue: 'mse' },
            LR: { type: Scratch.ArgumentType.NUMBER, defaultValue: 0.001 },
            DECAY: { type: Scratch.ArgumentType.NUMBER, defaultValue: 0.01 },
            BETA1: { type: Scratch.ArgumentType.NUMBER, defaultValue: 0.9 },
            BETA2: { type: Scratch.ArgumentType.NUMBER, defaultValue: 0.999 },
            EPS: { type: Scratch.ArgumentType.NUMBER, defaultValue: 1e-8 }
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
          items: [{ text: '均方误差(MSE)', value: 'mse' }, { text: '交叉熵', value: 'crossentropy' }]
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
          items: [{ text: '是', value: 'true' }, { text: '否', value: 'false' }]
        }
      }
    };
  }
}
Scratch.extensions.register(new EtudeTurboWarpML());