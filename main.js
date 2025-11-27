class WebGLCompute {
  constructor() {
    this.canvas = document.createElement('canvas');
    this.gl = this.canvas.getContext('webgl', { preserveDrawingBuffer: true }) || 
              this.canvas.getContext('experimental-webgl');
    
    if (!this.gl) {
      console.error("WebGL not supported");
      return;
    }

    const ext = this.gl.getExtension('OES_texture_float');
    if (!ext) console.warn("OES_texture_float not supported. Computation will fail.");
    this.gl.getExtension('WEBGL_color_buffer_float'); 

    this.programs = {
      multiply: this._createProgram(this._getVertexShader(), this._getMultiplyFragmentShader()),
      add: this._createProgram(this._getVertexShader(), this._getAddFragmentShader())
    };

    this.quadBuffer = this.gl.createBuffer();
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.quadBuffer);
    this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array([
      -1, -1,  1, -1, -1,  1,
      -1,  1,  1, -1,  1,  1
    ]), this.gl.STATIC_DRAW);
  }

  _getVertexShader() {
    return `
      attribute vec2 position;
      varying vec2 vTexCoord;
      void main() {
        vTexCoord = position * 0.5 + 0.5;
        gl_Position = vec4(position, 0.0, 1.0);
      }
    `;
  }

  _getMultiplyFragmentShader() {
    return `
      precision highp float;
      varying vec2 vTexCoord;
      uniform sampler2D uMatrixA;
      uniform sampler2D uMatrixB;
      uniform int uCommonDim;
      uniform vec2 uResolution; 

      void main() {
        vec2 pos = gl_FragCoord.xy;
        float row = floor(uResolution.y - pos.y);
        float col = floor(pos.x);

        float sum = 0.0;

        for (int k = 0; k < 10000; k++) {
          if (k >= uCommonDim) break;

          vec2 uvA = vec2((float(k) + 0.5) / float(uCommonDim), 1.0 - (row + 0.5) / uResolution.y);
          vec2 uvB = vec2((col + 0.5) / uResolution.x, 1.0 - (float(k) + 0.5) / float(uCommonDim));

          float valA = texture2D(uMatrixA, uvA).r;
          float valB = texture2D(uMatrixB, uvB).r;
          
          sum += valA * valB;
        }

        gl_FragColor = vec4(sum, 0.0, 0.0, 1.0);
      }
    `;
  }

  _getAddFragmentShader() {
    return `
      precision highp float;
      varying vec2 vTexCoord;
      uniform sampler2D uMatrixA;
      uniform sampler2D uMatrixB;

      void main() {
        float valA = texture2D(uMatrixA, vTexCoord).r;
        float valB = texture2D(uMatrixB, vTexCoord).r;
        gl_FragColor = vec4(valA + valB, 0.0, 0.0, 1.0);
      }
    `;
  }

  _createShader(type, source) {
    const shader = this.gl.createShader(type);
    this.gl.shaderSource(shader, source);
    this.gl.compileShader(shader);
    if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
      console.error(this.gl.getShaderInfoLog(shader));
      this.gl.deleteShader(shader);
      return null;
    }
    return shader;
  }

  _createProgram(vsSource, fsSource) {
    const program = this.gl.createProgram();
    const vs = this._createShader(this.gl.VERTEX_SHADER, vsSource);
    const fs = this._createShader(this.gl.FRAGMENT_SHADER, fsSource);
    this.gl.attachShader(program, vs);
    this.gl.attachShader(program, fs);
    this.gl.linkProgram(program);
    return program;
  }

  _createTexture(data, width, height) {
    const texture = this.gl.createTexture();
    this.gl.bindTexture(this.gl.TEXTURE_2D, texture);

    const flatData = new Float32Array(width * height * 4);
    let ptr = 0;
    for (let i = 0; i < height; i++) {
      for (let j = 0; j < width; j++) {
        const val = data[i][j];
        flatData[ptr] = val; 
        flatData[ptr+1] = 0;
        flatData[ptr+2] = 0;
        flatData[ptr+3] = 1;
        ptr += 4;
      }
    }

    this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA, width, height, 0, this.gl.RGBA, this.gl.FLOAT, flatData);
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.NEAREST);
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.NEAREST);
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
    return texture;
  }

  executeOperation(type, matrixA, matrixB) {
    if (!matrixA.length || !matrixB.length) return [];
    
    const rowsA = matrixA.length;
    const colsA = matrixA[0].length;
    
    let outWidth, outHeight, commonDim;
    
    if (type === 'multiply') {
      const rowsB = matrixB.length;
      const colsB = matrixB[0].length;
      if (colsA !== rowsB) return [];
      outWidth = colsB;
      outHeight = rowsA;
      commonDim = colsA;
    } else {
      if (rowsA !== matrixB.length || colsA !== matrixB[0].length) return [];
      outWidth = colsA;
      outHeight = rowsA;
      commonDim = 0;
    }

    this.canvas.width = outWidth;
    this.canvas.height = outHeight;
    this.gl.viewport(0, 0, outWidth, outHeight);

    const texA = this._createTexture(matrixA, colsA, rowsA);
    const texB = this._createTexture(matrixB, type === 'multiply' ? matrixB[0].length : colsA, matrixB.length);

    const targetTexture = this.gl.createTexture();
    this.gl.bindTexture(this.gl.TEXTURE_2D, targetTexture);
    this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA, outWidth, outHeight, 0, this.gl.RGBA, this.gl.FLOAT, null);
    
    const fbo = this.gl.createFramebuffer();
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, fbo);
    this.gl.framebufferTexture2D(this.gl.FRAMEBUFFER, this.gl.COLOR_ATTACHMENT0, this.gl.TEXTURE_2D, targetTexture, 0);

    const program = this.programs[type];
    this.gl.useProgram(program);

    const positionLocation = this.gl.getAttribLocation(program, "position");
    this.gl.enableVertexAttribArray(positionLocation);
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.quadBuffer);
    this.gl.vertexAttribPointer(positionLocation, 2, this.gl.FLOAT, false, 0, 0);

    this.gl.activeTexture(this.gl.TEXTURE0);
    this.gl.bindTexture(this.gl.TEXTURE_2D, texA);
    this.gl.uniform1i(this.gl.getUniformLocation(program, "uMatrixA"), 0);

    this.gl.activeTexture(this.gl.TEXTURE1);
    this.gl.bindTexture(this.gl.TEXTURE_2D, texB);
    this.gl.uniform1i(this.gl.getUniformLocation(program, "uMatrixB"), 1);

    if (type === 'multiply') {
       this.gl.uniform1i(this.gl.getUniformLocation(program, "uCommonDim"), commonDim);
       this.gl.uniform2f(this.gl.getUniformLocation(program, "uResolution"), outWidth, outHeight);
    }

    this.gl.drawArrays(this.gl.TRIANGLES, 0, 6);

    const rawBuffer = new Float32Array(outWidth * outHeight * 4);
    this.gl.readPixels(0, 0, outWidth, outHeight, this.gl.RGBA, this.gl.FLOAT, rawBuffer);

    this.gl.deleteTexture(texA);
    this.gl.deleteTexture(texB);
    this.gl.deleteTexture(targetTexture);
    this.gl.deleteFramebuffer(fbo);

    
    const result = [];
    for (let i = 0; i < outHeight; i++) {
      const row = [];

      const bufferRowIndex = (outHeight - 1 - i);
      
      for (let j = 0; j < outWidth; j++) {
        const ptr = (bufferRowIndex * outWidth + j) * 4;
        row.push(rawBuffer[ptr]);
      }
      result.push(row);
    }

    return result;
  }
}

const gpuCompute = new WebGLCompute();

class TensorMath {
  static Validators = {
    parseMatrix(jsonString) {
      try {
        const parsed = JSON.parse(jsonString);
        return (Array.isArray(parsed) && Array.isArray(parsed[0])) ? parsed : null;
      } catch {
        return null;
      }
    },
    
    isModelInitialized(state) {
      return state && state.isModelDefined;
    },

    validatePositiveInteger(value) {
      const num = parseInt(value, 10);
      return (isNaN(num) || num <= 0) ? null : num;
    }
  };

  static createZeros(shape) {
    if (shape.length === 1) return new Array(shape[0]).fill(0);
    return Array.from({ length: shape[0] }, () => new Array(shape[1]).fill(0));
  }

  static applyElementWise(target, source, operationFn, ...auxiliarySources) {
    if (!target || !source) return;
    const is2D = Array.isArray(target[0]);
    
    if (is2D) {
      for (let i = 0; i < target.length; i++) {
        for (let j = 0; j < target[i].length; j++) {
          const auxArgs = auxiliarySources.map(aux => aux ? aux[i][j] : undefined);
          target[i][j] = operationFn(target[i][j], source[i][j], ...auxArgs);
        }
      }
    } else {
      for (let i = 0; i < target.length; i++) {
        const auxArgs = auxiliarySources.map(aux => aux ? aux[i] : undefined);
        target[i] = operationFn(target[i], source[i], ...auxArgs);
      }
    }
  }

  static mapMatrix(matrixA, matrixB, operationFn) {
    if (!matrixA || !matrixB || matrixA.length !== matrixB.length || matrixA[0].length !== matrixB[0].length) {
      return [];
    }
    return matrixA.map((row, i) => row.map((val, j) => operationFn(val, matrixB[i][j])));
  }

  static transpose(matrix) {
    if (!matrix?.length || !matrix[0]) return [];
    return matrix[0].map((_, colIndex) => matrix.map(row => row[colIndex]));
  }

  static matrixMultiply(matrixA, matrixB) {
    if (!matrixA.length || !matrixB.length || matrixA[0].length !== matrixB.length) return [];
    return gpuCompute.executeOperation('multiply', matrixA, matrixB);
  }

  static sumRows(matrix) {
    return matrix?.[0] ? matrix[0].map((_, colIndex) => matrix.reduce((sum, row) => sum + row[colIndex], 0)) : [];
  }

  static matrixAdd(matrixA, matrixB) {
    return gpuCompute.executeOperation('add', matrixA, matrixB);
  }

  static Activations = {
    _functions: {
      relu: { forward: x => Math.max(0, x), backward: (x, grad) => (x > 0 ? grad : 0) },
      tanh: { forward: x => Math.tanh(x), backward: (x, grad) => (1 - x * x) * grad },
      sigmoid: { forward: x => 1 / (1 + Math.exp(-x)), backward: (x, grad) => (x * (1 - x)) * grad }
    },

    forward(matrix, type) {
      if (!matrix) return matrix;
      
      if (type === 'softmax') {
        return matrix.map(row => {
          const maxVal = Math.max(...row);
          const exps = row.map(v => Math.exp(v - maxVal));
          const sumExps = exps.reduce((a, b) => a + b, 0);
          return exps.map(e => e / sumExps);
        });
      }

      const act = this._functions[type];
      return act ? matrix.map(row => row.map(act.forward)) : matrix;
    },

    backward(activatedMatrix, type, gradients) {
      if (!activatedMatrix || !gradients) return gradients;

      if (type === 'softmax') {
        return activatedMatrix.map((row, i) => {
          const gradRow = gradients[i];
          return row.map((_, j) => {
            let sum = 0;
            for (let k = 0; k < row.length; k++) {
              sum += gradRow[k] * (j === k ? row[k] * (1 - row[k]) : -row[j] * row[k]);
            }
            return sum;
          });
        });
      }

      const act = this._functions[type];
      return act ? activatedMatrix.map((row, i) => row.map((val, j) => act.backward(val, gradients[i][j]))) : gradients;
    }
  };

  static Initializers = {
    generate(strategy, inputDim, outputDim) {
      let generatorFn;
      if (strategy === 'he') {
        generatorFn = () => (Math.random() - 0.5) * 2 * Math.sqrt(2.0 / inputDim);
      } else if (strategy === 'xavier') {
        generatorFn = () => (Math.random() - 0.5) * 2 * Math.sqrt(6 / (inputDim + outputDim));
      } else if (strategy === 'ones') {
        generatorFn = () => 1;
      } else {
        generatorFn = () => 0;
      }
      return Array.from({ length: outputDim }, () => Array.from({ length: inputDim }, generatorFn));
    }
  };

  static computeLossAndGradient(prediction, target, lossType) {
    if (!prediction || !target || prediction.length !== target.length) {
      return { loss: 0, gradients: [] };
    }
    
    const N = prediction.length;
    let totalLoss = 0;
    
    const gradients = prediction.map((row, i) => row.map((val, j) => {
      const targetVal = target[i][j];
      
      if (lossType === 'crossentropy') {
        const safeVal = val + 1e-7;
        totalLoss -= targetVal * Math.log(safeVal);
        return -targetVal / safeVal;
      } else {

        const diff = val - targetVal;
        totalLoss += diff * diff;
        return 2 * diff / N;
      }
    }));

    return { loss: totalLoss / N, gradients };
  }
}

class EtudeMLCore {
  constructor() {
    this._pendingLayerConfigs = [];
    this.globalState = this._initializeState();
  }

  _initializeState() {
    return {
      layers: [],
      isModelDefined: false,
      computationGraph: { forward: [], backward: [] },
      modelMeta: { 
        name: 'Etude-Model', 
        inputDim: null, 
        outputDim: null, 
        totalLayers: 0, 
        inputTensorName: 'tensor_0', 
        createdAt: Date.now() 
      },
      parameters: {},
      gradients: {},
      forwardCache: {},
      optimizerState: { step: 0, firstMoment: {}, secondMoment: {} }
    };
  }

  _createZeroGradients(layerConfig) {
    const isLinear = layerConfig.type === 'linear';
    return {
      weight: TensorMath.createZeros(isLinear ? [layerConfig.outputDim, layerConfig.inputDim] : [layerConfig.inputDim]),
      bias: layerConfig.useBias ? TensorMath.createZeros(isLinear ? [layerConfig.outputDim] : [layerConfig.inputDim]) : null
    };
  }

  _queueLayer(config) { 
    this._pendingLayerConfigs.push(config); 
  }

  addLinearLayer(args) {
    const inputDim = TensorMath.Validators.validatePositiveInteger(args.INPUT_DIM);
    const outputDim = TensorMath.Validators.validatePositiveInteger(args.OUTPUT_DIM);
    if (inputDim && outputDim) {
      this._queueLayer({ 
        type: 'linear', 
        inputDim: inputDim, 
        outputDim: outputDim, 
        useBias: args.USE_BIAS === 'true' 
      });
    }
  }

  addActivationLayer(args) { 
    this._queueLayer({ type: 'activation', activation: args.ACTIVATION }); 
  }

  addLayerNorm(args) { 
    this._queueLayer({ type: 'layernorm', useBias: args.USE_BIAS === 'true' }); 
  }

  addRMSNorm() { 
    this._queueLayer({ type: 'rmsnorm' }); 
  }

  compileModel(args) {
    if (!this._pendingLayerConfigs.length) return;

    const firstLinearLayer = this._pendingLayerConfigs.find(l => l.type === 'linear');
    if (!firstLinearLayer) return;

    let currentDimCheck = null;
    for (const layer of this._pendingLayerConfigs) {
      if (layer.type === 'linear') {
        if (currentDimCheck !== null && layer.inputDim !== currentDimCheck) {
          console.warn("Dimension mismatch in model definition");
          return this.resetModel();
        }
        currentDimCheck = layer.outputDim;
      }
    }

    const inputDimension = firstLinearLayer.inputDim;
    this.globalState = this._initializeState();
    this.globalState.modelMeta.inputDim = inputDimension;
    let currentDimension = inputDimension;

    this._pendingLayerConfigs.forEach((config, index) => {
      const inputName = index === 0 ? 'tensor_0' : `tensor_${index}`;
      const outputName = `tensor_${index + 1}`;
      const layerId = `layer_${index}_${config.type}`;
      let outputDimension = currentDimension;
      
      if (config.type === 'linear') outputDimension = config.outputDim;

      const fullConfig = { 
        ...config, 
        id: layerId, 
        inputDim: currentDimension, 
        outputDim: outputDimension, 
        inputName, 
        outputName,
        activation: config.activation || 'none'
      };
      
      this.globalState.layers.push(fullConfig);

      if (config.type === 'linear') {
        this.globalState.computationGraph.forward.push({ 
          id: `op_${index}`, 
          type: 'linear', 
          layerId, 
          inputs: [inputName], 
          outputs: [outputName] 
        });
        
        this.globalState.parameters[layerId] = {
          weight: TensorMath.Initializers.generate(args.INIT || 'he', currentDimension, outputDimension),
          bias: config.useBias ? new Array(outputDimension).fill(0) : null
        };
        currentDimension = outputDimension;

      } else if (config.type === 'activation') {
        this.globalState.computationGraph.forward.push({ 
          id: `op_${index}`, 
          type: 'activation', 
          activationType: config.activation, 
          inputs: [inputName], 
          outputs: [outputName] 
        });

      } else {
        const isRMS = config.type === 'rmsnorm';
        this.globalState.computationGraph.forward.push({ 
          id: `op_${index}`, 
          type: config.type, 
          layerId, 
          inputs: [inputName], 
          outputs: [outputName] 
        });
        
        this.globalState.parameters[layerId] = {
          weight: new Array(currentDimension).fill(1),
          bias: (config.useBias && !isRMS) ? new Array(currentDimension).fill(0) : null
        };
      }
      
      this.globalState.gradients[layerId] = this._createZeroGradients(fullConfig);
    });

    this.globalState.modelMeta.totalLayers = this.globalState.layers.length;
    this.globalState.modelMeta.outputDim = currentDimension;
    this.globalState.isModelDefined = true;
    this._pendingLayerConfigs = [];
  }

  forwardPropagate(args) {
    if (!TensorMath.Validators.isModelInitialized(this.globalState)) return '[]';
    
    const inputTensor = TensorMath.Validators.parseMatrix(args.INPUT);
    if (!inputTensor || inputTensor[0].length !== this.globalState.modelMeta.inputDim) return '[]';

    this.globalState.forwardCache = {};
    const initialInputName = this.globalState.computationGraph.forward[0]?.inputs[0] || 'tensor_0';

    this.globalState.forwardCache[initialInputName] = { preActivation: null, postActivation: inputTensor };
    
    let currentTensor = inputTensor;

    for (const node of this.globalState.computationGraph.forward) {
      let outputTensor;

      if (node.type === 'linear') {
        const params = this.globalState.parameters[node.layerId];

        const weightTransposed = TensorMath.transpose(params.weight);
        const matMulResult = TensorMath.matrixMultiply(currentTensor, weightTransposed);

        
        if (params.bias) {
             const biasMatrix = matMulResult.map(() => params.bias); // Replicate bias rows
             outputTensor = TensorMath.matrixAdd(matMulResult, biasMatrix);
        } else {
             outputTensor = matMulResult;
        }

        this.globalState.forwardCache[node.outputs[0]] = { preActivation: currentTensor, postActivation: outputTensor };

      } else if (node.type === 'activation') {
        outputTensor = TensorMath.Activations.forward(currentTensor, node.activationType);
        this.globalState.forwardCache[node.outputs[0]] = { preActivation: currentTensor, postActivation: outputTensor };

      } else {
        outputTensor = this._forwardNormalization(node, currentTensor);
      }
      currentTensor = outputTensor;
    }
    return JSON.stringify(currentTensor);
  }

  _forwardNormalization(node, input) {
    const params = this.globalState.parameters[node.layerId];
    const isRMS = node.type === 'rmsnorm';
    const normCache = []; 

    const output = input.map(row => {
      let mean = 0;
      let invStd = 0;

      if (isRMS) {
        const sumSquares = row.reduce((acc, val) => acc + val * val, 0);
        invStd = 1 / Math.sqrt(sumSquares / row.length + 1e-5);
        normCache.push({ invStd });
      } else {
        mean = row.reduce((acc, val) => acc + val, 0) / row.length;
        const variance = row.reduce((acc, val) => acc + (val - mean) ** 2, 0) / row.length;
        invStd = 1 / Math.sqrt(variance + 1e-5);
        normCache.push({ mean, invStd });
      }

      return row.map((val, i) => {
        const normalizedVal = isRMS ? val * invStd : (val - mean) * invStd;
        return normalizedVal * params.weight[i] + (params.bias ? params.bias[i] : 0);
      });
    });

    this.globalState.forwardCache[node.outputs[0]] = { 
      preActivation: input, 
      postActivation: output, 
      statistics: normCache 
    };
    return output;
  }

  getModelStructure() {
    if (!this.globalState.isModelDefined) return JSON.stringify({ error: 'Model not defined' });
    return JSON.stringify({
      format: 'etude-ml-model-1.3', 
      meta: this.globalState.modelMeta,
      layers: this.globalState.layers.map(l => ({ 
        config: l, 
        parameters: this.globalState.parameters[l.id] 
      })),
      computationGraph: this.globalState.computationGraph
    });
  }

  loadModel(args) {
    try {
      const data = JSON.parse(args.JSON);
      if (!data.format?.startsWith('etude-ml')) return;
      
      this.resetModel();
      this.globalState.modelMeta = data.meta;
      this.globalState.computationGraph = data.computationGraph;
      this.globalState.isModelDefined = true;
      
      (data.layers || []).forEach(({ config, parameters }) => {
        this.globalState.layers.push(config);
        if (parameters) this.globalState.parameters[config.id] = parameters;
        this.globalState.gradients[config.id] = this._createZeroGradients(config);
      });
    } catch (e) { 
      console.warn("Model load failed", e); 
    }
  }

  resetModel() { 
    this.globalState = this._initializeState(); 
    this._pendingLayerConfigs = []; 
  }

  isModelReady() { 
    return this.globalState.isModelDefined; 
  }
}

class AutogradEngine {
  constructor(core) { 
    this.core = core; 
  }

  zeroGradients() {
    this.core.globalState.layers.forEach(layer => {
      this.core.globalState.gradients[layer.id] = this.core._createZeroGradients(layer);
    });
  }

  backwardPropagate(args) {
    if (!TensorMath.Validators.isModelInitialized(this.core.globalState)) return;
    
    const outputGradient = TensorMath.Validators.parseMatrix(args.GRAD);
    if (!outputGradient) return;

    const graph = this.core.globalState.computationGraph;

    const gradientMap = { [graph.forward.at(-1).outputs[0]]: outputGradient };

    for (let i = graph.forward.length - 1; i >= 0; i--) {
      const node = graph.forward[i];
      const dy = gradientMap[node.outputs[0]]; 
      if (!dy) continue;

      if (node.type === 'linear') {
        const fwdData = this.core.globalState.forwardCache[node.outputs[0]];
        const params = this.core.globalState.parameters[node.layerId];

        gradientMap[node.inputs[0]] = TensorMath.matrixMultiply(dy, params.weight);

        this.core.globalState.gradients[node.layerId] = {
          weight: TensorMath.matrixMultiply(TensorMath.transpose(dy), fwdData.preActivation),
          bias: params.bias ? TensorMath.sumRows(dy) : null
        };

      } else if (node.type === 'activation') {
        const fwdData = this.core.globalState.forwardCache[node.outputs[0]];

        gradientMap[node.inputs[0]] = TensorMath.Activations.backward(fwdData.postActivation, node.activationType, dy);

      } else if (node.type.includes('norm')) {
        this._backwardNormalization(node, dy, gradientMap, node.type === 'rmsnorm');
      }
    }
  }

  _backwardNormalization(node, dy, gradientMap, isRMS) {
    const { preActivation: input, statistics } = this.core.globalState.forwardCache[node.outputs[0]];
    const { weight: gamma, bias: beta } = this.core.globalState.parameters[node.layerId];
    const [BatchSize, FeatureDim] = [input.length, input[0].length];
    
    const dGamma = new Array(FeatureDim).fill(0);
    const dBeta = beta ? new Array(FeatureDim).fill(0) : null;
    const dx = [];

    for (let i = 0; i < BatchSize; i++) {
      const { invStd, mean } = statistics[i];
      const rowInput = input[i];
      const rowDy = dy[i];
      
      const xHat = rowInput.map(val => (val - (isRMS ? 0 : mean)) * invStd);

      for (let j = 0; j < FeatureDim; j++) {
        dGamma[j] += rowDy[j] * xHat[j];
        if (beta) dBeta[j] += rowDy[j];
      }

      const dL_dxHat = rowDy.map((val, j) => val * gamma[j]);
      
      let rowDx;
      if (isRMS) {
        const sumProd = dL_dxHat.reduce((acc, val, j) => acc + val * rowInput[j], 0);
        const factor = (invStd * invStd) / FeatureDim * sumProd;
        rowDx = dL_dxHat.map((val, j) => invStd * (val - rowInput[j] * factor));
      } else {
        const sumGrad = dL_dxHat.reduce((acc, val) => acc + val, 0);
        const sumProd = dL_dxHat.reduce((acc, val, j) => acc + val * xHat[j], 0);
        rowDx = dL_dxHat.map((val, j) => 
          (invStd / FeatureDim) * (FeatureDim * val - sumGrad - xHat[j] * sumProd)
        );
      }
      dx.push(rowDx);
    }
    
    gradientMap[node.inputs[0]] = dx;
    this.core.globalState.gradients[node.layerId] = { weight: dGamma, bias: dBeta };
  }
}

class OptimizerEngine {
  constructor(core, autograd) {
    this.core = core;
    this.autograd = autograd;
  }

  stepSGD(args) {
    const learningRate = parseFloat(args.LR) || 0.01;
    this._performOptimizationStep(args, (param, grad) => param - learningRate * grad, false);
  }

  stepAdamW(args) {
    this._performOptimizationStep(args, null, true);
  }
  
  _performOptimizationStep(args, simpleUpdateKernel, requiresComplexState = false) {
    if (!TensorMath.Validators.isModelInitialized(this.core.globalState)) return;
    
    const prediction = TensorMath.Validators.parseMatrix(args.PRED);
    const target = TensorMath.Validators.parseMatrix(args.TARGET);
    if (!prediction || !target) return;

    this.autograd.zeroGradients();

    const { gradients: outputGrads } = TensorMath.computeLossAndGradient(prediction, target, args.LOSS || 'mse');

    this.autograd.backwardPropagate({ GRAD: JSON.stringify(outputGrads) });

    const optState = this.core.globalState.optimizerState;
    optState.step++;

    let adamConfig = null;
    if (requiresComplexState && !simpleUpdateKernel) {
        const t = optState.step;
        const beta1 = parseFloat(args.BETA1) || 0.9;
        const beta2 = parseFloat(args.BETA2) || 0.999;
        adamConfig = {
            lr: parseFloat(args.LR) || 0.001,
            beta1: beta1,
            beta2: beta2,
            epsilon: parseFloat(args.EPS) || 1e-8,
            decay: parseFloat(args.DECAY) || 0.01,
            biasCorrection1: 1 - Math.pow(beta1, t),
            biasCorrection2: 1 - Math.pow(beta2, t)
        };
    }

    this.core.globalState.layers.forEach(layer => {
      const layerId = layer.id;
      const params = this.core.globalState.parameters[layerId];
      const grads = this.core.globalState.gradients[layerId];
      if (!params) return;

      ['weight', 'bias'].forEach(paramKey => {
        if (!params[paramKey]) return;
        
        const paramTensor = params[paramKey];
        const gradTensor = grads[paramKey];
        
        if (simpleUpdateKernel) {
             TensorMath.applyElementWise(paramTensor, gradTensor, simpleUpdateKernel);
        } else {
             if (!optState.firstMoment[layerId]) optState.firstMoment[layerId] = {};
             if (!optState.secondMoment[layerId]) optState.secondMoment[layerId] = {};

             if (!optState.firstMoment[layerId][paramKey]) {
                 const shape = Array.isArray(paramTensor[0]) ? [paramTensor.length, paramTensor[0].length] : [paramTensor.length];
                 optState.firstMoment[layerId][paramKey] = TensorMath.createZeros(shape);
                 optState.secondMoment[layerId][paramKey] = TensorMath.createZeros(shape);
             }

             const mTensor = optState.firstMoment[layerId][paramKey];
             const vTensor = optState.secondMoment[layerId][paramKey];
             
             const updateAdamKernel = (w, g, m, v) => {

                 const nextM = adamConfig.beta1 * m + (1 - adamConfig.beta1) * g;

                 const nextV = adamConfig.beta2 * v + (1 - adamConfig.beta2) * g * g;

                 const mHat = nextM / adamConfig.biasCorrection1;
                 const vHat = nextV / adamConfig.biasCorrection2;

                 const nextW = w - adamConfig.lr * ((mHat / (Math.sqrt(vHat) + adamConfig.epsilon)) + adamConfig.decay * w);
                 
                 return { w: nextW, m: nextM, v: nextV };
             };

             if (Array.isArray(paramTensor[0])) { 
                 for(let i=0; i < paramTensor.length; i++) {
                     for(let j=0; j < paramTensor[i].length; j++) {
                         const result = updateAdamKernel(paramTensor[i][j], gradTensor[i][j], mTensor[i][j], vTensor[i][j]);
                         paramTensor[i][j] = result.w; 
                         mTensor[i][j] = result.m; 
                         vTensor[i][j] = result.v;
                     }
                 }
             } else { 
                 for(let i=0; i < paramTensor.length; i++) {
                     const result = updateAdamKernel(paramTensor[i], gradTensor[i], mTensor[i], vTensor[i]);
                     paramTensor[i] = result.w; 
                     mTensor[i] = result.m; 
                     vTensor[i] = result.v;
                 }
             }
        }
      });
    });
  }
}

class LinearAlgebraExtension {
  multiply(args) {
    const matrixA = TensorMath.Validators.parseMatrix(args.A);
    const matrixB = TensorMath.Validators.parseMatrix(args.B);
    return JSON.stringify(TensorMath.matrixMultiply(matrixA, matrixB));
  }
  
  add(args) {
    const matrixA = TensorMath.Validators.parseMatrix(args.A);
    const matrixB = TensorMath.Validators.parseMatrix(args.B);
    return JSON.stringify(TensorMath.matrixAdd(matrixA, matrixB));
  }
}

class EtudeMLExtension {
  constructor() {
    this.core = new EtudeMLCore();
    this.autograd = new AutogradEngine(this.core);
    this.optimizer = new OptimizerEngine(this.core, this.autograd);
    this.math = new LinearAlgebraExtension();

    this._bindMethods(this.core);
    this._bindMethods(this.optimizer);
    this._bindMethods(this.math);

    this.endModelDefinition = this.core.compileModel.bind(this.core);
    this.forward = this.core.forwardPropagate.bind(this.core);
    this.clearModel = this.core.resetModel.bind(this.core);
    this.isModelDefined = this.core.isModelReady.bind(this.core);

    this.matrixMultiplication = this.math.multiply.bind(this.math);
    this.matrixAddition = this.math.add.bind(this.math);
  }

  _bindMethods(module) {
    const proto = Object.getPrototypeOf(module);
    Object.getOwnPropertyNames(proto)
      .filter(name => name !== 'constructor' && typeof module[name] === 'function' && !name.startsWith('_'))
      .forEach(name => {
        if (!this[name]) {
           this[name] = module[name].bind(module);
        }
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
      version: '0.1.0',
      blocks: [
        {
          blockType: Scratch.BlockType.LABEL,
          text: '模型构建与管理'
        },
        {
          opcode: 'endModelDefinition',
          blockType: Scratch.BlockType.COMMAND,
          text: '构建并初始化模型 策略 [INIT]',
          arguments: {
            INIT: {
              type: Scratch.ArgumentType.STRING,
              menu: 'INIT_MENU',
              defaultValue: 'he'
            }
          }
        },
        {
          opcode: 'addLinearLayer',
          blockType: Scratch.BlockType.COMMAND,
          text: '添加线性层 输入维度 [INPUT_DIM] 输出维度 [OUTPUT_DIM] 使用偏置 [USE_BIAS]',
          arguments: {
            INPUT_DIM: {
              type: Scratch.ArgumentType.NUMBER,
              defaultValue: 4
            },
            OUTPUT_DIM: {
              type: Scratch.ArgumentType.NUMBER,
              defaultValue: 4
            },
            USE_BIAS: {
              type: Scratch.ArgumentType.STRING,
              menu: 'BOOL_MENU',
              defaultValue: 'true'
            }
          }
        },
        {
          opcode: 'addActivationLayer',
          blockType: Scratch.BlockType.COMMAND,
          text: '添加激活函数 [ACTIVATION]',
          arguments: {
            ACTIVATION: {
              type: Scratch.ArgumentType.STRING,
              menu: 'ACTIVATION_MENU',
              defaultValue: 'relu'
            }
          }
        },
        {
          opcode: 'addLayerNorm',
          blockType: Scratch.BlockType.COMMAND,
          text: '添加层归一化 (LayerNorm) 使用偏置 [USE_BIAS]',
          arguments: {
            USE_BIAS: {
              type: Scratch.ArgumentType.STRING,
              menu: 'BOOL_MENU',
              defaultValue: 'true'
            }
          }
        },
        {
          opcode: 'addRMSNorm',
          blockType: Scratch.BlockType.COMMAND,
          text: '添加RMS归一化 (RMSNorm)'
        },
        {
          opcode: 'loadModel',
          blockType: Scratch.BlockType.COMMAND,
          text: '加载模型 (JSON) [JSON]',
          arguments: {
            JSON: {
              type: Scratch.ArgumentType.STRING,
              defaultValue: '{}'
            }
          }
        },
        {
          opcode: 'getModelStructure',
          blockType: Scratch.BlockType.REPORTER,
          text: '导出模型 (JSON)'
        },
        {
          opcode: 'clearModel',
          blockType: Scratch.BlockType.COMMAND,
          text: '清除当前模型'
        },
        {
          opcode: 'isModelDefined',
          blockType: Scratch.BlockType.BOOLEAN,
          text: '模型已加载?'
        },

        {
          blockType: Scratch.BlockType.LABEL,
          text: '推理与训练'
        },
        {
          opcode: 'forward',
          blockType: Scratch.BlockType.REPORTER,
          text: '推理 输入向量 [INPUT]',
          arguments: {
            INPUT: {
              type: Scratch.ArgumentType.STRING,
              defaultValue: '[[1, 1]]'
            }
          }
        },
        {
          opcode: 'stepSGD',
          blockType: Scratch.BlockType.COMMAND,
          text: 'SGD优化 预测 [PRED] 目标 [TARGET] 损失 [LOSS] LR [LR]',
          arguments: {
            PRED: {
              type: Scratch.ArgumentType.STRING,
              defaultValue: '[[0]]'
            },
            TARGET: {
              type: Scratch.ArgumentType.STRING,
              defaultValue: '[[1]]'
            },
            LOSS: {
              type: Scratch.ArgumentType.STRING,
              menu: 'LOSS_MENU',
              defaultValue: 'mse'
            },
            LR: {
              type: Scratch.ArgumentType.NUMBER,
              defaultValue: 0.01
            }
          }
        },
        {
          opcode: 'stepAdamW',
          blockType: Scratch.BlockType.COMMAND,
          text: 'AdamW优化 预测 [PRED] 目标 [TARGET] 损失 [LOSS] LR [LR] Decay [DECAY]',
          arguments: {
            PRED: {
              type: Scratch.ArgumentType.STRING,
              defaultValue: '[[0]]'
            },
            TARGET: {
              type: Scratch.ArgumentType.STRING,
              defaultValue: '[[1]]'
            },
            LOSS: {
              type: Scratch.ArgumentType.STRING,
              menu: 'LOSS_MENU',
              defaultValue: 'mse'
            },
            LR: {
              type: Scratch.ArgumentType.NUMBER,
              defaultValue: 0.001
            },
            DECAY: {
              type: Scratch.ArgumentType.NUMBER,
              defaultValue: 0.01
            },
            BETA1: {
              type: Scratch.ArgumentType.NUMBER,
              defaultValue: 0.9
            },
            BETA2: {
              type: Scratch.ArgumentType.NUMBER,
              defaultValue: 0.999
            },
            EPS: {
              type: Scratch.ArgumentType.NUMBER,
              defaultValue: 1e-8
            }
          }
        },

        {
          blockType: Scratch.BlockType.LABEL,
          text: '线性代数 (WebGL)'
        },
        {
          opcode: 'matrixMultiplication',
          blockType: Scratch.BlockType.REPORTER,
          text: '矩阵 [A] × [B]',
          arguments: {
            A: {
              type: Scratch.ArgumentType.STRING,
              defaultValue: '[[1,2],[3,4]]'
            },
            B: {
              type: Scratch.ArgumentType.STRING,
              defaultValue: '[[5,6],[7,8]]'
            }
          }
        },
        {
          opcode: 'matrixAddition',
          blockType: Scratch.BlockType.REPORTER,
          text: '矩阵 [A] + [B]',
          arguments: {
            A: {
              type: Scratch.ArgumentType.STRING,
              defaultValue: '[[1,2],[3,4]]'
            },
            B: {
              type: Scratch.ArgumentType.STRING,
              defaultValue: '[[5,6],[7,8]]'
            }
          }
        }
      ],
      menus: {
        ACTIVATION_MENU: {
          items: ['relu', 'tanh', 'sigmoid', 'softmax'].map(v => ({
            text: v.charAt(0).toUpperCase() + v.slice(1),
            value: v
          }))
        },
        LOSS_MENU: {
          items: [
            { text: '均方误差(MSE)', value: 'mse' },
            { text: '交叉熵', value: 'crossentropy' }
          ]
        },
        INIT_MENU: {
          items: [
            { text: 'He (ReLU推荐)', value: 'he' },
            { text: 'Xavier (Sigmoid/Tanh)', value: 'xavier' },
            { text: '全零', value: 'zeros' },
            { text: '全一', value: 'ones' }
          ]
        },
        BOOL_MENU: {
          items: [
            { text: '是', value: 'true' },
            { text: '否', value: 'false' }
          ]
        }
      }
    };
  }
}
Scratch.extensions.register(new EtudeMLExtension());