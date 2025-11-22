class autograd {
  constructor() {
    this.gradBuffer = {}; // 存储中间梯度
  }

  getInfo() {
    return {
      id: 'EtudeTurboWarpMLAutograd',
      name: 'Etude-TurboWarp-ML_Autograd',
      color1: '#FF6B6B',
      author: 'Asuka | Lin Xin',
      version: '1.0.1',
      blocks: [
        {
          opcode: 'backward',
          func: 'backward',
          blockType: Scratch.BlockType.COMMAND,
          text: '对输出梯度 [GRAD] 执行反向传播',
          arguments: {
            GRAD: { type: Scratch.ArgumentType.STRING, defaultValue: '[[1,0]]' }
          }
        },
        {
          opcode: 'getParamGrad',
          func: 'getParamGrad',
          blockType: Scratch.BlockType.REPORTER,
          text: '获取 [PARAM] 的梯度',
          arguments: {
            PARAM: { type: Scratch.ArgumentType.STRING, menu: 'PARAM_MENU' }
          }
        },
        {
          opcode: 'zeroGrad',
          func: 'zeroGrad',
          blockType: Scratch.BlockType.COMMAND,
          text: '清零所有梯度'
        }
      ],
      menus: {
        PARAM_MENU: this._getParamMenu.bind(this)
      }
    };
  }

  // 获取参数列表用于菜单
  _getParamMenu() {
    try {
      const coreExt = Scratch.extensions.getExtension('EtudeTurboWarpML');
      if (!coreExt || !coreExt.globalState.gradients) return [];
      
      return Object.keys(coreExt.globalState.gradients).map(key => ({
        text: key, value: key
      }));
    } catch(e) {
      return [];
    }
  }

  backward(args) {
    const coreExt = Scratch.extensions.getExtension('EtudeTurboWarpML');
    if (!coreExt || !coreExt.globalState.isModelDefined) {
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

    // 从最后一层开始反向传播
    const tape = [...coreExt.globalState.computationGraph.forward].reverse();
    this.gradBuffer = {};
    
    // 最后一层的梯度
    const lastNode = tape[0];
    if (lastNode.type === 'activation') {
      this.gradBuffer[lastNode.inputs[0]] = outputGrad;
    } else {
      this.gradBuffer[lastNode.outputs[0]] = outputGrad;
    }

    // 遍历计算图反向传播
    for (const node of tape) {
      if (node.type === 'linear') {
        this._linearBackward(node, coreExt);
      } else if (node.type === 'activation') {
        this._activationBackward(node, coreExt);
      }
    }
    
    console.log('[autograd] 反向传播完成');
  }

  _linearBackward(node, coreExt) {
    const inputName = node.inputs[0];
    const outputName = node.outputs[0];
    const outputGrad = this.gradBuffer[outputName];
    
    if (!outputGrad) return;

    const layerId = node.params[0].split('.')[0];
    const layer = coreExt.globalState.layers.find(l => l.id === layerId);
    const inputData = coreExt.globalState.forwardData?.[inputName];
    
    if (!layer || !inputData) return;

    const inputGrad = this._matMul(outputGrad, this._transpose(layer.parameters.weight.data));
    this.gradBuffer[inputName] = inputGrad;

    // 计算权重梯度: dL/dW = X^T * dL/dY
    const weightGrad = this._matMul(this._transpose(inputData), outputGrad);
    
    // 计算偏置梯度: dL/db = sum(dL/dY, axis=0)
    const biasGrad = this._sumRows(outputGrad);
    
    // 存储梯度
    coreExt.globalState.gradients[layerId] = {
      weight: weightGrad,
      bias: biasGrad
    };
  }

  _activationBackward(node, coreExt) {
    const inputName = node.inputs[0];
    const outputName = node.outputs[0];
    const outputGrad = this.gradBuffer[outputName];
    
    if (!outputGrad) return;

    const inputData = coreExt.globalState.forwardData?.[inputName];
    if (!inputData) return;

    let inputGrad;
    switch(node.activation_type) {
      case 'relu':
        inputGrad = inputData.map(row => 
          row.map(val => val > 0 ? 1 : 0)
        );
        break;
      case 'tanh':
        // tanh' = 1 - tanh^2
        inputGrad = inputData.map(row => 
          row.map(val => 1 - val * val)
        );
        break;
      case 'sigmoid':
        // sigmoid' = sigmoid * (1 - sigmoid)
        inputGrad = inputData.map(row => 
          row.map(val => val * (1 - val))
        );
        break;
      case 'softmax':
        // 简化处理，假设与交叉熵组合
        inputGrad = outputGrad;
        break;
      default:
        inputGrad = outputGrad;
    }
    
    // 逐元素相乘
    this.gradBuffer[inputName] = this._hadamard(inputGrad, outputGrad);
  }

  getParamGrad(args) {
    const coreExt = Scratch.extensions.getExtension('EtudeTurboWarpML');
    if (!coreExt) return '[]';
    
    const grad = coreExt.globalState.gradients[args.PARAM];
    return JSON.stringify(grad || { weight: [], bias: [] });
  }

  zeroGrad() {
    const coreExt = Scratch.extensions.getExtension('EtudeTurboWarpML');
    if (!coreExt) return;
    
    const layers = coreExt.globalState.layers;
    layers.forEach(layer => {
      const id = layer.id;
      const inDim = layer.input_dim;
      const outDim = layer.output_dim;
      
      coreExt.globalState.gradients[id] = {
        weight: Array(outDim).fill().map(() => Array(inDim).fill(0)),
        bias: Array(outDim).fill(0)
      };
    });
    
    console.log('[autograd] 梯度已清零');
  }

  // 矩阵转置
  _transpose(matrix) {
    return matrix[0].map((_, col) => matrix.map(row => row[col]));
  }

  // 矩阵乘法
  _matMul(a, b) {
    if (!a || !b || a[0].length !== b.length) return [];
    return a.map(row => 
      b[0].map((_, j) => 
        row.reduce((sum, val, k) => sum + val * b[k][j], 0)
      )
    );
  }

  // 行求和
  _sumRows(matrix) {
    return matrix[0].map((_, col) => 
      matrix.reduce((sum, row) => sum + row[col], 0)
    );
  }

  // Hadamard积
  _hadamard(a, b) {
    return a.map((row, i) => row.map((val, j) => val * b[i][j]));
  }
}

Scratch.extensions.register(new autograd());