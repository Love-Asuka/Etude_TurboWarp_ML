class optimizer {
  constructor() {
    this.learningRate = 0.01;
  }

  getInfo() {
    return {
      id: 'EtudeTurboWarpMLOptimizer',
      name: 'Etude-TurboWarp-ML_Optimizer',
      color1: '#4ECDC4',
      author: 'Asuka | Lin Xin',
      version: '1.0.1',
      blocks: [
        {
          opcode: 'setLearningRate',
          func: 'setLearningRate',
          blockType: Scratch.BlockType.COMMAND,
          text: '设置学习率 [LR]',
          arguments: {
            LR: { type: Scratch.ArgumentType.NUMBER, defaultValue: 0.01 }
          }
        },
        {
          opcode: 'stepSGD',
          func: 'stepSGD',
          blockType: Scratch.BlockType.COMMAND,
          text: '执行SGD优化步骤'
        }
      ]
    };
  }

  setLearningRate(args) {
    this.learningRate = parseFloat(args.LR) || 0.01;
    console.log(`[optimizer] 学习率设置为: ${this.learningRate}`);
  }

  stepSGD() {
    const coreExt = Scratch.extensions.getExtension('EtudeTurboWarpML');
    const autogradExt = Scratch.extensions.getExtension('EtudeTurboWarpMLAutograd');
    
    if (!coreExt || !coreExt.globalState.isModelDefined) {
      console.error('[optimizer] 模型未定义');
      return;
    }

    const layers = coreExt.globalState.layers;
    const gradients = coreExt.globalState.gradients;
    const params = coreExt.globalState.parameters;

    layers.forEach(layer => {
      const layerId = layer.id;
      const layerGrad = gradients[layerId];
      const layerParams = params[layerId];
      
      if (!layerGrad || !layerParams) return;

      // 更新权重: W = W - lr * dW
      layerParams.weight = layerParams.weight.map((row, i) =>
        row.map((val, j) => val - this.learningRate * layerGrad.weight[i][j])
      );
      
      // 更新偏置: b = b - lr * db
      layerParams.bias = layerParams.bias.map((val, i) =>
        val - this.learningRate * layerGrad.bias[i]
      );
    });

    console.log('[optimizer] SGD步骤完成');
  }
}

Scratch.extensions.register(new optimizer());