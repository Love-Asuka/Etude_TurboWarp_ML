const MLUtils = {
  Validation: {
    parseMatrix(str) {
      try {
        const parsed = JSON.parse(str);
        return (Array.isArray(parsed) && Array.isArray(parsed[0])) ? parsed : null;
      } catch { return null; }
    },
    ensureModel(state) { return state && state.isModelDefined; },
    validatePositiveInt(val) {
      const num = parseInt(val);
      return (isNaN(num) || num <= 0) ? null : num;
    }
  },

  zeros(shape) {
    if (shape.length === 1) return new Array(shape[0]).fill(0);
    return Array.from({ length: shape[0] }, () => new Array(shape[1]).fill(0));
  },

  tensorApply(target, source, fn, ...auxSources) {
    if (!target || !source) return;
    const is2D = Array.isArray(target[0]);
    
    if (is2D) {
      for (let i = 0; i < target.length; i++) {
        for (let j = 0; j < target[i].length; j++) {
          const auxArgs = auxSources.map(aux => aux ? aux[i][j] : undefined);
          target[i][j] = fn(target[i][j], source[i][j], ...auxArgs);
        }
      }
    } else {
      for (let i = 0; i < target.length; i++) {
        const auxArgs = auxSources.map(aux => aux ? aux[i] : undefined);
        target[i] = fn(target[i], source[i], ...auxArgs);
      }
    }
  },

  mapMatrices(a, b, op) {
    if (!a || !b || a.length !== b.length || a[0].length !== b[0].length) return [];
    return a.map((row, i) => row.map((val, j) => op(val, b[i][j])));
  },

  transpose(m) {
    if (!m?.length || !m[0]) return [];
    return m[0].map((_, c) => m.map(r => r[c]));
  },

  matMul(a, b) {
    if (!a.length || !b.length || a[0].length !== b.length) return [];
    const bT = this.transpose(b);
    return a.map(row => bT.map(col => row.reduce((s, v, k) => s + v * col[k], 0)));
  },

  sumRows(m) {
    return m?.[0] ? m[0].map((_, c) => m.reduce((s, r) => s + r[c], 0)) : [];
  },

  matAdd(a, b) { return this.mapMatrices(a, b, (x, y) => x + y); },

  ActivationRegistry: {
    _acts: {
      relu: { f: x => Math.max(0, x), b: (x, g) => (x > 0 ? g : 0) },
      tanh: { f: x => Math.tanh(x), b: (x, g) => (1 - x * x) * g },
      sigmoid: { f: x => 1 / (1 + Math.exp(-x)), b: (x, g) => (x * (1 - x)) * g }
    },

    apply(m, type) {
      if (!m) return m;
      if (type === 'softmax') {
        return m.map(row => {
          const max = Math.max(...row);
          const exps = row.map(v => Math.exp(v - max));
          const sum = exps.reduce((a, b) => a + b, 0);
          return exps.map(e => e / sum);
        });
      }
      const act = this._acts[type];
      return act ? m.map(r => r.map(act.f)) : m;
    },

    derivative(activated, type, grad) {
      if (!activated || !grad) return grad;
      if (type === 'softmax') {
        return activated.map((row, i) => {
          const g = grad[i];
          return row.map((_, j) => {
            let sum = 0;
            for (let k = 0; k < row.length; k++) {
              sum += g[k] * (j === k ? row[k] * (1 - row[k]) : -row[j] * row[k]);
            }
            return sum;
          });
        });
      }
      const act = this._acts[type];
      return act ? activated.map((r, i) => r.map((v, j) => act.b(v, grad[i][j]))) : grad;
    }
  },

  Initializers: {
    generate(strategy, inDim, outDim) {
      let fn;
      if (strategy === 'he') fn = () => (Math.random() - 0.5) * 2 * Math.sqrt(2.0 / inDim);
      else if (strategy === 'xavier') fn = () => (Math.random() - 0.5) * 2 * Math.sqrt(6 / (inDim + outDim));
      else if (strategy === 'ones') fn = () => 1;
      else fn = () => 0;
      return Array.from({ length: outDim }, () => Array.from({ length: inDim }, fn));
    }
  },

  computeLossAndGradient(pred, target, lossType) {
    if (!pred || !target || pred.length !== target.length) return { loss: 0, grad: [] };
    const N = pred.length;
    let totalLoss = 0;
    
    const grad = pred.map((row, i) => row.map((val, j) => {
      const t = target[i][j];
      if (lossType === 'crossentropy') {
        const safeVal = val + 1e-7;
        totalLoss -= t * Math.log(safeVal);
        return -t / safeVal;
      } else {
        const diff = val - t;
        totalLoss += diff * diff;
        return 2 * diff / N;
      }
    }));

    return { loss: totalLoss / N, grad };
  }
};

class EtudeTurboWarpMLCore {
  constructor() {
    this._pendingLayers = [];
    this.globalState = this._createFreshState();
  }

  _createFreshState() {
    return {
      layers: [], isModelDefined: false,
      computationGraph: { forward: [], backward: [] },
      modelMeta: { name: 'Etude-Model', inputDim: null, outputDim: null, totalLayers: 0, inputName: 'tensor_0', created: Date.now() },
      parameters: {}, gradients: {}, forwardData: {},
      optimizerState: { step: 0, m: {}, v: {} }
    };
  }

  _generateZeroGrads(layer) {
    const isLinear = layer.type === 'linear';
    return {
      weight: MLUtils.zeros(isLinear ? [layer.output_dim, layer.input_dim] : [layer.input_dim]),
      bias: layer.use_bias ? MLUtils.zeros(isLinear ? [layer.output_dim] : [layer.input_dim]) : null
    };
  }

  _addPending(config) { this._pendingLayers.push(config); }

  addLinearLayer(args) {
    const inD = MLUtils.Validation.validatePositiveInt(args.INPUT_DIM);
    const outD = MLUtils.Validation.validatePositiveInt(args.OUTPUT_DIM);
    if (inD && outD) this._addPending({ type: 'linear', input_dim: inD, output_dim: outD, use_bias: args.USE_BIAS === 'true' });
  }

  addActivationLayer(args) { this._addPending({ type: 'activation', activation: args.ACTIVATION }); }
  addLayerNorm(args) { this._addPending({ type: 'layernorm', use_bias: args.USE_BIAS === 'true' }); }
  addRMSNorm() { this._addPending({ type: 'rmsnorm' }); }

  endModelDefinition(args) {
    if (!this._pendingLayers.length) return;
    const firstLin = this._pendingLayers.find(l => l.type === 'linear');
    if (!firstLin) return;

    // 验证连续性
    let dimCheck = null;
    for (const l of this._pendingLayers) {
      if (l.type === 'linear') {
        if (dimCheck !== null && l.input_dim !== dimCheck) return this.clearModel();
        dimCheck = l.output_dim;
      }
    }

    const inputDim = firstLin.input_dim;
    this.globalState = this._createFreshState();
    this.globalState.modelMeta.inputDim = inputDim;
    let currDim = inputDim;

    this._pendingLayers.forEach((cfg, idx) => {
      const inName = idx === 0 ? 'tensor_0' : `tensor_${idx}`;
      const outName = `tensor_${idx + 1}`;
      const layId = `layer_${idx}_${cfg.type}`;
      let outDim = currDim;
      
      if (cfg.type === 'linear') outDim = cfg.output_dim;

      const fullCfg = { ...cfg, id: layId, input_dim: currDim, output_dim: outDim, input_name: inName, output_name: outName };
      fullCfg.activation = cfg.activation || 'none';
      
      this.globalState.layers.push(fullCfg);

      // 构建计算图与参数
      if (cfg.type === 'linear') {
        this.globalState.computationGraph.forward.push({ id: `op_${idx}`, type: 'linear', layerId: layId, inputs: [inName], outputs: [outName] });
        this.globalState.parameters[layId] = {
          weight: MLUtils.Initializers.generate(args.INIT || 'he', currDim, outDim),
          bias: cfg.use_bias ? new Array(outDim).fill(0) : null
        };
        currDim = outDim;
      } else if (cfg.type === 'activation') {
        this.globalState.computationGraph.forward.push({ id: `op_${idx}`, type: 'activation', activation_type: cfg.activation, inputs: [inName], outputs: [outName] });
      } else {
        const isRMS = cfg.type === 'rmsnorm';
        this.globalState.computationGraph.forward.push({ id: `op_${idx}`, type: cfg.type, layerId: layId, inputs: [inName], outputs: [outName] });
        this.globalState.parameters[layId] = {
          weight: new Array(currDim).fill(1),
          bias: (cfg.use_bias && !isRMS) ? new Array(currDim).fill(0) : null
        };
      }
      this.globalState.gradients[layId] = this._generateZeroGrads(fullCfg);
    });

    this.globalState.modelMeta.totalLayers = this.globalState.layers.length;
    this.globalState.modelMeta.outputDim = currDim;
    this.globalState.isModelDefined = true;
    this._pendingLayers = [];
  }

  forward(args) {
    if (!MLUtils.Validation.ensureModel(this.globalState)) return '[]';
    const input = MLUtils.Validation.parseMatrix(args.INPUT);
    if (!input || input[0].length !== this.globalState.modelMeta.inputDim) return '[]';

    this.globalState.forwardData = {};
    const firstIn = this.globalState.computationGraph.forward[0]?.inputs[0] || 'tensor_0';
    this.globalState.forwardData[firstIn] = { preActivation: null, postActivation: input };
    
    let val = input;
    for (const node of this.globalState.computationGraph.forward) {
      let outVal;
      if (node.type === 'linear') {
        const p = this.globalState.parameters[node.layerId];
        outVal = val.map(r => p.weight.map((wR, i) => r.reduce((s, v, k) => s + v * wR[k], 0) + (p.bias ? p.bias[i] : 0)));
        this.globalState.forwardData[node.outputs[0]] = { preActivation: val, postActivation: outVal };
      } else if (node.type === 'activation') {
        outVal = MLUtils.ActivationRegistry.apply(val, node.activation_type);
        this.globalState.forwardData[node.outputs[0]] = { preActivation: val, postActivation: outVal };
      } else {
        outVal = this._normForward(node, val);
      }
      val = outVal;
    }
    return JSON.stringify(val);
  }

  _normForward(node, input) {
    const params = this.globalState.parameters[node.layerId];
    const isRMS = node.type === 'rmsnorm';
    const cache = [];
    const output = input.map(row => {
      let mean = 0, inv = 0;
      if (isRMS) {
        inv = 1 / Math.sqrt(row.reduce((a, b) => a + b * b, 0) / row.length + 1e-5);
        cache.push({ inv });
      } else {
        mean = row.reduce((a, b) => a + b, 0) / row.length;
        inv = 1 / Math.sqrt(row.reduce((a, b) => a + (b - mean) ** 2, 0) / row.length + 1e-5);
        cache.push({ mean, inv });
      }
      return row.map((v, i) => {
        const norm = isRMS ? v * inv : (v - mean) * inv;
        return norm * params.weight[i] + (params.bias ? params.bias[i] : 0);
      });
    });
    this.globalState.forwardData[node.outputs[0]] = { preActivation: input, postActivation: output, cache };
    return output;
  }

  getModelStructure() {
    if (!this.globalState.isModelDefined) return JSON.stringify({ error: 'Model not defined' });
    return JSON.stringify({
      format: 'etude-ml-model-1.3', meta: this.globalState.modelMeta,
      layers: this.globalState.layers.map(l => ({ config: l, parameters: this.globalState.parameters[l.id] })),
      computation_graph: this.globalState.computationGraph
    });
  }

  loadModel(args) {
    try {
      const data = JSON.parse(args.JSON);
      if (!data.format?.startsWith('etude-ml')) return;
      this.clearModel();
      this.globalState.modelMeta = data.meta;
      this.globalState.computationGraph = data.computation_graph;
      this.globalState.isModelDefined = true;
      (data.layers || []).forEach(({ config, parameters }) => {
        this.globalState.layers.push(config);
        if (parameters) this.globalState.parameters[config.id] = parameters;
        this.globalState.gradients[config.id] = this._generateZeroGrads(config);
      });
    } catch (e) { console.warn("Load failed", e); }
  }

  clearModel() { this.globalState = this._createFreshState(); this._pendingLayers = []; }
  isModelDefined() { return this.globalState.isModelDefined; }
}

class EtudeTurboWarpMLAutograd {
  constructor(core) { this.core = core; }

  zeroGrad() {
    this.core.globalState.layers.forEach(l => this.core.globalState.gradients[l.id] = this.core._generateZeroGrads(l));
  }

  backward(args) {
    if (!MLUtils.Validation.ensureModel(this.core.globalState)) return;
    const gradIn = MLUtils.Validation.parseMatrix(args.GRAD);
    if (!gradIn) return;

    const graph = this.core.globalState.computationGraph;
    const gradMap = { [graph.forward.at(-1).outputs[0]]: gradIn };

    for (let i = graph.forward.length - 1; i >= 0; i--) {
      const node = graph.forward[i];
      const dy = gradMap[node.outputs[0]];
      if (!dy) continue;

      if (node.type === 'linear') {
        const fwd = this.core.globalState.forwardData[node.outputs[0]];
        const p = this.core.globalState.parameters[node.layerId];
        gradMap[node.inputs[0]] = MLUtils.matMul(dy, p.weight);
        this.core.globalState.gradients[node.layerId] = {
          weight: MLUtils.matMul(MLUtils.transpose(dy), fwd.preActivation),
          bias: p.bias ? MLUtils.sumRows(dy) : null
        };
      } else if (node.type === 'activation') {
        const fwd = this.core.globalState.forwardData[node.outputs[0]];
        gradMap[node.inputs[0]] = MLUtils.ActivationRegistry.derivative(fwd.postActivation, node.activation_type, dy);
      } else if (node.type.includes('norm')) {
        this._normBackward(node, dy, gradMap, node.type === 'rmsnorm');
      }
    }
  }

  _normBackward(node, dy, gradMap, isRMS) {
    const { preActivation: x, cache } = this.core.globalState.forwardData[node.outputs[0]];
    const { weight: gamma, bias: beta } = this.core.globalState.parameters[node.layerId];
    const [N, D] = [x.length, x[0].length];
    
    const dGamma = new Array(D).fill(0);
    const dBeta = beta ? new Array(D).fill(0) : null;
    const dx = [];

    for (let i = 0; i < N; i++) {
      const { inv, mean } = cache[i];
      const r_x = x[i];
      const r_dy = dy[i];
      const x_hat = r_x.map(v => (v - (isRMS ? 0 : mean)) * inv);

      for (let j = 0; j < D; j++) {
        dGamma[j] += r_dy[j] * x_hat[j];
        if (beta) dBeta[j] += r_dy[j];
      }

      const dldx_hat = r_dy.map((v, j) => v * gamma[j]);
      
      let row_dx;
      if (isRMS) {
        const sum = dldx_hat.reduce((a, v, j) => a + v * r_x[j], 0);
        const fac = (inv * inv) / D * sum;
        row_dx = dldx_hat.map((v, j) => inv * (v - r_x[j] * fac));
      } else {
        const sum1 = dldx_hat.reduce((a, b) => a + b, 0);
        const sum2 = dldx_hat.reduce((a, v, j) => a + v * x_hat[j], 0);
        row_dx = dldx_hat.map((v, j) => (inv / D) * (D * v - sum1 - x_hat[j] * sum2));
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

  _performOptimization(args, updateKernel, needsState = false) {
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

      const applyUpdate = (key) => {
        if (!params[key]) return;
        
        let m = null, v = null;
        if (needsState) {
          if (!optState.m[pid]) optState.m[pid] = {};
          if (!optState.v[pid]) optState.v[pid] = {};
          
          // Lazy init state
          if (!optState.m[pid][key]) {
            const shape = Array.isArray(params[key][0]) ? [params[key].length, params[key][0].length] : [params[key].length];
            optState.m[pid][key] = MLUtils.zeros(shape);
            optState.v[pid][key] = MLUtils.zeros(shape);
          }
          m = optState.m[pid][key];
          v = optState.v[pid][key];
        }

        MLUtils.tensorApply(params[key], grads[key], updateKernel, m, v);
      };

      applyUpdate('weight');
      applyUpdate('bias');
    });
  }

  stepSGD(args) {
    const lr = parseFloat(args.LR) || 0.01;
    this._performOptimization(args, (w, g) => w - lr * g, false);
  }

  stepAdamW(args) {
    const conf = {
      lr: parseFloat(args.LR) || 0.001,
      b1: parseFloat(args.BETA1) || 0.9,
      b2: parseFloat(args.BETA2) || 0.999,
      eps: parseFloat(args.EPS) || 1e-8,
      decay: parseFloat(args.DECAY) || 0.01
    };
    
    const t = this.core.globalState.optimizerState.step + 1; 
    const c1 = 1 - Math.pow(conf.b1, t);
    const c2 = 1 - Math.pow(conf.b2, t);


    const kernel = (w, g, m, v) => {

      const nm = conf.b1 * m + (1 - conf.b1) * g;
      const nv = conf.b2 * v + (1 - conf.b2) * g * g;
      return w; 
    };
    

    this._performOptimization(args, null, true);
  }
  

  _performOptimization(args, simpleKernel, needsState = false) {
    if (!MLUtils.Validation.ensureModel(this.core.globalState)) return;
    const pred = MLUtils.Validation.parseMatrix(args.PRED);
    const target = MLUtils.Validation.parseMatrix(args.TARGET);
    if (!pred || !target) return;

    this.autograd.zeroGrad();
    const { grad } = MLUtils.computeLossAndGradient(pred, target, args.LOSS || 'mse');
    this.autograd.backward({ GRAD: JSON.stringify(grad) });

    const optState = this.core.globalState.optimizerState;
    optState.step++;

    let adamConf = null;
    if (needsState && !simpleKernel) {
        const t = optState.step;
        adamConf = {
            lr: parseFloat(args.LR) || 0.001,
            b1: parseFloat(args.BETA1) || 0.9,
            b2: parseFloat(args.BETA2) || 0.999,
            eps: parseFloat(args.EPS) || 1e-8,
            decay: parseFloat(args.DECAY) || 0.01,
            c1: 1 - Math.pow(parseFloat(args.BETA1)||0.9, t),
            c2: 1 - Math.pow(parseFloat(args.BETA2)||0.999, t)
        };
    }

    this.core.globalState.layers.forEach(layer => {
      const pid = layer.id;
      const params = this.core.globalState.parameters[pid];
      const grads = this.core.globalState.gradients[pid];
      if (!params) return;

      ['weight', 'bias'].forEach(key => {
        if (!params[key]) return;
        const pT = params[key];
        const gT = grads[key];
        
        if (simpleKernel) {

             MLUtils.tensorApply(pT, gT, simpleKernel);
        } else {

             if (!optState.m[pid]) optState.m[pid] = {};
             if (!optState.v[pid]) optState.v[pid] = {};
             if (!optState.m[pid][key]) {
                 const shape = Array.isArray(pT[0]) ? [pT.length, pT[0].length] : [pT.length];
                 optState.m[pid][key] = MLUtils.zeros(shape);
                 optState.v[pid][key] = MLUtils.zeros(shape);
             }
             const mT = optState.m[pid][key];
             const vT = optState.v[pid][key];
             
             const update = (w, g, m, v) => {
                 const nm = adamConf.b1 * m + (1 - adamConf.b1) * g;
                 const nv = adamConf.b2 * v + (1 - adamConf.b2) * g * g;
                 const mHat = nm / adamConf.c1;
                 const vHat = nv / adamConf.c2;
                 const nw = w - adamConf.lr * ((mHat / (Math.sqrt(vHat) + adamConf.eps)) + adamConf.decay * w);
                 return { w: nw, m: nm, v: nv };
             };

             if (Array.isArray(pT[0])) { // 2D
                 for(let i=0; i<pT.length; i++) {
                     for(let j=0; j<pT[i].length; j++) {
                         const res = update(pT[i][j], gT[i][j], mT[i][j], vT[i][j]);
                         pT[i][j] = res.w; mT[i][j] = res.m; vT[i][j] = res.v;
                     }
                 }
             } else { // 1D
                 for(let i=0; i<pT.length; i++) {
                     const res = update(pT[i], gT[i], mT[i], vT[i]);
                     pT[i] = res.w; mT[i] = res.m; vT[i] = res.v;
                 }
             }
        }
      });
    });
  }
}

class EtudeTurboWarpMLLinearAlgebra {
  matrixMultiplication(args) {
    const a = MLUtils.Validation.parseMatrix(args.A);
    const b = MLUtils.Validation.parseMatrix(args.B);
    return JSON.stringify(MLUtils.matMul(a, b));
  }
  matrixAddition(args) {
    const a = MLUtils.Validation.parseMatrix(args.A);
    const b = MLUtils.Validation.parseMatrix(args.B);
    return JSON.stringify(MLUtils.matAdd(a, b));
  }
}

class EtudeTurboWarpML {
  constructor() {
    this.core = new EtudeTurboWarpMLCore();
    this.autograd = new EtudeTurboWarpMLAutograd(this.core);
    this.optimizer = new EtudeTurboWarpMLOptimizer(this.core, this.autograd);
    this.linearAlgebra = new EtudeTurboWarpMLLinearAlgebra();
    
    [this.core, this.optimizer, this.linearAlgebra].forEach(mod => {
      Object.getOwnPropertyNames(Object.getPrototypeOf(mod))
        .filter(n => n !== 'constructor' && typeof mod[n] === 'function' && !n.startsWith('_'))
        .forEach(n => this[n] = (...args) => mod[n](...args));
    });
  }

  getInfo() {
    return {
      id: 'EtudeTurboWarpML', name: 'Etude-TurboWarp-ML',
      color1: '#4C97FF', color2: '#3d85c6', color3: '#2e5d8f',
      author: 'Asuka | Lin Xi', version: '0.0.9',
      blocks: [
        { blockType: Scratch.BlockType.LABEL, text: '模型构建与管理' },
        { opcode: 'endModelDefinition', blockType: Scratch.BlockType.COMMAND, text: '构建并初始化模型 策略 [INIT]', arguments: { INIT: { type: Scratch.ArgumentType.STRING, menu: 'INIT_MENU', defaultValue: 'he' } } },
        { opcode: 'addLinearLayer', blockType: Scratch.BlockType.COMMAND, text: '添加线性层 输入维度 [INPUT_DIM] 输出维度 [OUTPUT_DIM] 使用偏置 [USE_BIAS]', arguments: { INPUT_DIM: { type: Scratch.ArgumentType.NUMBER, defaultValue: 4 }, OUTPUT_DIM: { type: Scratch.ArgumentType.NUMBER, defaultValue: 4 }, USE_BIAS: { type: Scratch.ArgumentType.STRING, menu: 'BOOL_MENU', defaultValue: 'true' } } },
        { opcode: 'addActivationLayer', blockType: Scratch.BlockType.COMMAND, text: '添加激活函数 [ACTIVATION]', arguments: { ACTIVATION: { type: Scratch.ArgumentType.STRING, menu: 'ACTIVATION_MENU', defaultValue: 'relu' } } },
        { opcode: 'addLayerNorm', blockType: Scratch.BlockType.COMMAND, text: '添加层归一化 (LayerNorm) 使用偏置 [USE_BIAS]', arguments: { USE_BIAS: { type: Scratch.ArgumentType.STRING, menu: 'BOOL_MENU', defaultValue: 'true' } } },
        { opcode: 'addRMSNorm', blockType: Scratch.BlockType.COMMAND, text: '添加RMS归一化 (RMSNorm)' },
        { opcode: 'loadModel', blockType: Scratch.BlockType.COMMAND, text: '加载模型 (JSON) [JSON]', arguments: { JSON: { type: Scratch.ArgumentType.STRING, defaultValue: '{}' } } },
        { opcode: 'getModelStructure', blockType: Scratch.BlockType.REPORTER, text: '导出模型 (JSON)' },
        { opcode: 'clearModel', blockType: Scratch.BlockType.COMMAND, text: '清除当前模型' },
        { opcode: 'isModelDefined', blockType: Scratch.BlockType.BOOLEAN, text: '模型已加载?' },
        { blockType: Scratch.BlockType.LABEL, text: '推理与训练' },
        { opcode: 'forward', blockType: Scratch.BlockType.REPORTER, text: '推理 输入向量 [INPUT]', arguments: { INPUT: { type: Scratch.ArgumentType.STRING, defaultValue: '[[1, 1]]' } } },
        { opcode: 'stepSGD', blockType: Scratch.BlockType.COMMAND, text: 'SGD优化 预测 [PRED] 目标 [TARGET] 损失 [LOSS] LR [LR]', arguments: { PRED: { type: Scratch.ArgumentType.STRING, defaultValue: '[[0]]' }, TARGET: { type: Scratch.ArgumentType.STRING, defaultValue: '[[1]]' }, LOSS: { type: Scratch.ArgumentType.STRING, menu: 'LOSS_MENU', defaultValue: 'mse' }, LR: { type: Scratch.ArgumentType.NUMBER, defaultValue: 0.01 } } },
        { opcode: 'stepAdamW', blockType: Scratch.BlockType.COMMAND, text: 'AdamW优化 预测 [PRED] 目标 [TARGET] 损失 [LOSS] LR [LR] Decay [DECAY]', arguments: { PRED: { type: Scratch.ArgumentType.STRING, defaultValue: '[[0]]' }, TARGET: { type: Scratch.ArgumentType.STRING, defaultValue: '[[1]]' }, LOSS: { type: Scratch.ArgumentType.STRING, menu: 'LOSS_MENU', defaultValue: 'mse' }, LR: { type: Scratch.ArgumentType.NUMBER, defaultValue: 0.001 }, DECAY: { type: Scratch.ArgumentType.NUMBER, defaultValue: 0.01 }, BETA1: { type: Scratch.ArgumentType.NUMBER, defaultValue: 0.9 }, BETA2: { type: Scratch.ArgumentType.NUMBER, defaultValue: 0.999 }, EPS: { type: Scratch.ArgumentType.NUMBER, defaultValue: 1e-8 } } },
        { blockType: Scratch.BlockType.LABEL, text: '线性代数' },
        { opcode: 'matrixMultiplication', blockType: Scratch.BlockType.REPORTER, text: '矩阵 [A] × [B]', arguments: { A: { type: Scratch.ArgumentType.STRING, defaultValue: '[[1,2],[3,4]]' }, B: { type: Scratch.ArgumentType.STRING, defaultValue: '[[5,6],[7,8]]' } } },
        { opcode: 'matrixAddition', blockType: Scratch.BlockType.REPORTER, text: '矩阵 [A] + [B]', arguments: { A: { type: Scratch.ArgumentType.STRING, defaultValue: '[[1,2],[3,4]]' }, B: { type: Scratch.ArgumentType.STRING, defaultValue: '[[5,6],[7,8]]' } } }
      ],
      menus: {
        ACTIVATION_MENU: { items: ['relu', 'tanh', 'sigmoid', 'softmax'].map(v => ({ text: v.charAt(0).toUpperCase() + v.slice(1), value: v })) },
        LOSS_MENU: { items: [{ text: '均方误差(MSE)', value: 'mse' }, { text: '交叉熵', value: 'crossentropy' }] },
        INIT_MENU: { items: [{ text: 'He (ReLU推荐)', value: 'he' }, { text: 'Xavier (Sigmoid/Tanh)', value: 'xavier' }, { text: '全零', value: 'zeros' }, { text: '全一', value: 'ones' }] },
        BOOL_MENU: { items: [{ text: '是', value: 'true' }, { text: '否', value: 'false' }] }
      }
    };
  }
}
Scratch.extensions.register(new EtudeTurboWarpML());