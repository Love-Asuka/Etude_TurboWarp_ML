class core {
  constructor() {
    // 初始化全局状态
    this.globalState = {
      layers: [],              // 存储所有层配置
      currentInputDim: null,   // 当前层输入维度（即上一层输出维度）
      isModelDefined: false,   // 模型是否定义完成
      computationGraph: {
        forward: [],           // 前向计算图
        backward: []           // 反向计算图（留待autograd实现）
      },
      modelMeta: {
        name: '未命名模型',
        inputDim: null,
        outputDim: null,
        totalLayers: 0,
        created: null
      },
      // 新增：用于自动微分和优化的状态
      parameters: {},          // 存储模型参数（权重和偏置）
      gradients: {},           // 存储参数梯度
      forwardData: {}          // 存储前向传播中间结果
    };
  }

  getInfo() {
    return {
      id: 'EtudeTurboWarpML',
      name: 'Etude-TurboWarp-ML',
      color1: '#4C97FF',
      author: 'Asuka | Lin Xin',
      version: '1.0.1',
      blocks: [
        {
          opcode: 'startModelDefinition',
          func: 'startModelDefinition',
          blockType: Scratch.BlockType.COMMAND,
          text: '开始模型定义，输入维度 [INPUT_DIM]',
          arguments: {
            INPUT_DIM: {
              type: Scratch.ArgumentType.NUMBER,
              defaultValue: 4
            }
          }
        },
        {
          opcode: 'addLinearLayer',
          func: 'addLinearLayer',
          blockType: Scratch.BlockType.COMMAND,
          text: '添加线性层 输出维度 [OUTPUT_DIM] 激活函数 [ACTIVATION]',
          arguments: {
            OUTPUT_DIM: {
              type: Scratch.ArgumentType.NUMBER,
              defaultValue: 8
            },
            ACTIVATION: {
              type: Scratch.ArgumentType.STRING,
              menu: 'ACTIVATION_MENU',
              defaultValue: 'relu'
            }
          }
        },
        {
          opcode: 'endModelDefinition',
          func: 'endModelDefinition',
          blockType: Scratch.BlockType.COMMAND,
          text: '结束模型定义'
        },
        {
          opcode: 'getModelStructure',
          func: 'getModelStructure',
          blockType: Scratch.BlockType.REPORTER,
          text: '获取模型结构JSON'
        },
        {
          opcode: 'clearModel',
          func: 'clearModel',
          blockType: Scratch.BlockType.COMMAND,
          text: '清除模型'
        },
        {
          opcode: 'isModelDefined',
          func: 'isModelDefined',
          blockType: Scratch.BlockType.BOOLEAN,
          text: '模型是否已定义'
        }
      ],
      menus: {
        ACTIVATION_MENU: [
          { text: 'ReLU', value: 'relu' },
          { text: 'Tanh', value: 'tanh' },
          { text: 'Sigmoid', value: 'sigmoid' },
          { text: 'Softmax', value: 'softmax' },
          { text: '无', value: 'none' }
        ]
      }
    };
  }

  // 开始定义模型，设置输入维度
  startModelDefinition(args) {
    const inputDim = parseInt(args.INPUT_DIM);
    if (isNaN(inputDim) || inputDim <= 0) {
      console.error('[core.js] 错误：输入维度必须是正整数');
      return;
    }

    // 重置状态
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
      // 重置新增状态
      parameters: {},
      gradients: {},
      forwardData: {}
    };

    console.log(`[core.js] 模型定义开始，输入维度: ${inputDim}`);
  }

  // 添加线性层
  addLinearLayer(args) {
    if (this.globalState.currentInputDim === null) {
      console.error('[core.js] 错误：请先调用"开始模型定义"');
      return;
    }

    const outputDim = parseInt(args.OUTPUT_DIM);
    const activation = args.ACTIVATION;

    if (isNaN(outputDim) || outputDim <= 0) {
      console.error('[core.js] 错误：输出维度必须是正整数');
      return;
    }

    const layerIndex = this.globalState.layers.length;
    const layerId = `layer_${layerIndex}_linear`;
    const graphId = `op_${String(layerIndex + 1).padStart(3, '0')}`;

    // 创建层配置（与权重示例.json格式兼容）
    const layerConfig = {
      id: layerId,
      type: 'linear',
      input_dim: this.globalState.currentInputDim,
      output_dim: outputDim,
      activation: activation,
      graph_id: graphId
    };

    this.globalState.layers.push(layerConfig);

    // 构建前向计算图
    const inputName = layerIndex === 0 ? 'input_vector' : `op_${String(layerIndex).padStart(3, '0')}_input`;
    const linearOutputName = `activation_${layerIndex + 1}`;

    // 添加线性运算节点
    this.globalState.computationGraph.forward.push({
      id: graphId,
      type: 'linear',
      inputs: [inputName],
      outputs: [linearOutputName],
      params: [`${layerId}.weight`, `${layerId}.bias`]
    });

    // 添加激活函数节点（如果有）
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

    // 初始化该层的梯度存储空间（用于自动微分）
    this.globalState.gradients[layerId] = {
      weight: Array(outputDim).fill().map(() => Array(this.globalState.currentInputDim).fill(0)),
      bias: Array(outputDim).fill(0)
    };

    // 更新状态
    this.globalState.currentInputDim = outputDim;
    this.globalState.modelMeta.totalLayers = this.globalState.layers.length;
    this.globalState.modelMeta.outputDim = outputDim;

    console.log(`[core.js] 添加线性层: ${JSON.stringify(layerConfig)}`);
  }

  // 结束模型定义
  endModelDefinition() {
    if (this.globalState.layers.length === 0) {
      console.error('[core.js] 错误：模型中至少需要一个层');
      return;
    }

    this.globalState.isModelDefined = true;
    
    // 修正最终输出节点命名
    const forwardGraph = this.globalState.computationGraph.forward;
    if (forwardGraph.length > 0) {
      const lastNode = forwardGraph[forwardGraph.length - 1];
      if (lastNode.type === 'activation') {
        lastNode.outputs = ['final_output'];
      } else if (lastNode.type === 'linear') {
        // 如果最后一层没有激活函数，修正线性层输出
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

    // 初始化模型参数和清空缓存数据
    this.globalState.parameters = {};
    this.globalState.forwardData = {};

    // 为每一层初始化参数（权重和偏置）
    this.globalState.layers.forEach((layer) => {
      const inDim = layer.input_dim;
      const outDim = layer.output_dim;
      const layerId = layer.id;
      
      // 权重：小随机数初始化（He初始化简化版，适用于ReLU）
      const scale = Math.sqrt(2.0 / inDim);
      this.globalState.parameters[layerId] = {
        weight: Array(outDim).fill().map(() => 
          Array(inDim).fill().map(() => (Math.random() - 0.5) * 2 * scale)
        ),
        bias: Array(outDim).fill(0) // 偏置初始化为零
      };
    });

    console.log('[core.js] 模型定义完成，参数已初始化');
    console.log('[core.js] 模型结构:', this.globalState);
  }

  // 获取模型结构JSON（与权重示例.json格式一致）
  getModelStructure() {
    if (!this.globalState.isModelDefined) {
      return JSON.stringify({ error: '模型尚未定义' }, null, 2);
    }

    const structure = {
      format: 'turbowarp-nn-weights',
      version: '1.0',
      model_meta: {
        name: this.globalState.modelMeta.name,
        total_layers: this.globalState.modelMeta.totalLayers,
        input_dim: this.globalState.modelMeta.inputDim,
        output_dim: this.globalState.modelMeta.outputDim,
        created: this.globalState.modelMeta.created
      },
      layers: this.globalState.layers.map(layer => ({
        id: layer.id,
        type: layer.type,
        input_dim: layer.input_dim,
        output_dim: layer.output_dim,
        activation: layer.activation,
        graph_id: layer.graph_id,
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
    };

    return JSON.stringify(structure, null, 2);
  }

  // 清除模型
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
    console.log('[core.js] 模型已清除');
  }

  // 检查模型是否已定义
  isModelDefined() {
    return this.globalState.isModelDefined;
  }
}

// 注册扩展
Scratch.extensions.register(new core());