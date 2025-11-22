class NeuralNetworkExtension {
    constructor() {
        this.modelState = {
            layers: [],
            computationGraph: null,
            isCompiled: false,
            forwardCache: {},
            gradients: {},
            parameterMap: {}
        };
    }

    getInfo() {
        return {
            id: 'EtudeTurboWarpcore',
            name: 'Etude-TurboWarp-ML',
            author: 'Asuka | Lin Xin',
            version: '1.2.0', // 版本更新
            blocks: [
                {
                    opcode: 'define_linear',
                    blockType: Scratch.BlockType.COMMAND,
                    text: '构建: 添加线性层 [NAME] 输入 [IN] 输出 [OUT] 激活 [ACT]',
                    arguments: {
                        NAME: { type: Scratch.ArgumentType.STRING, defaultValue: 'fc1' },
                        IN: { type: Scratch.ArgumentType.NUMBER, defaultValue: 4 },
                        OUT: { type: Scratch.ArgumentType.NUMBER, defaultValue: 8 },
                        ACT: { type: Scratch.ArgumentType.STRING, menu: 'activationMenu' }
                    }
                },
                {
                    opcode: 'compile_model',
                    blockType: Scratch.BlockType.COMMAND,
                    text: '构建: 编译模型'
                },
                {
                    opcode: 'load_weights_json',
                    blockType: Scratch.BlockType.COMMAND,
                    text: '核心: 从JSON加载完整模型 [JSON]',
                    arguments: {
                        JSON: { type: Scratch.ArgumentType.STRING, defaultValue: '{}' }
                    }
                },
                {
                    opcode: 'forward_pass',
                    blockType: Scratch.BlockType.REPORTER,
                    text: '运行: 前向传播 输入 [INPUT]',
                    arguments: {
                        INPUT: { type: Scratch.ArgumentType.STRING, defaultValue: '[1,0,0,0]' }
                    }
                },
                {
                    opcode: 'backward_pass',
                    blockType: Scratch.BlockType.COMMAND,
                    text: '训练: 反向传播 目标 [TARGET]',
                    arguments: {
                        TARGET: { type: Scratch.ArgumentType.STRING, defaultValue: '[0,1]' }
                    }
                },
                {
                    opcode: 'apply_gradients',
                    blockType: Scratch.BlockType.COMMAND,
                    text: '训练: 应用梯度 学习率 [LR]',
                    arguments: {
                        LR: { type: Scratch.ArgumentType.NUMBER, defaultValue: 0.01 }
                    }
                },
                {
                    opcode: 'get_output',
                    blockType: Scratch.BlockType.REPORTER,
                    text: '数据: 获取当前输出'
                },
                {
                    opcode: 'export_weights',
                    blockType: Scratch.BlockType.REPORTER,
                    text: '数据: 导出模型JSON'
                },
                {
                    opcode: 'matrix_multiply',
                    blockType: Scratch.BlockType.REPORTER,
                    text: '工具: 矩阵乘法 [A] × [B]',
                    arguments: {
                        A: { type: Scratch.ArgumentType.STRING, defaultValue: '[[1,2],[3,4]]' },
                        B: { type: Scratch.ArgumentType.STRING, defaultValue: '[[5,6],[7,8]]' }
                    }
                }
            ],
            menus: {
                activationMenu: ['relu', 'tanh', 'softmax', 'sigmoid', 'none']
            }
        };
    }

    // ============= 模型定义 =============

    define_linear(args) {
        if (this.modelState.isCompiled) {
            console.warn('Warning: Cannot add layers to a compiled model.');
            return;
        }

        let inputDim = parseInt(args.IN);
        const outputDim = parseInt(args.OUT);

        if (this.modelState.layers.length > 0) {
            const lastLayer = this.modelState.layers[this.modelState.layers.length - 1];
            inputDim = lastLayer.output_dim;
        }

        const layerIndex = this.modelState.layers.length;
        const layerId = `layer_${layerIndex}_linear`;
        
        const weightMatrix = this.initializeWeight(outputDim, inputDim);
        const biasVector = new Array(outputDim).fill(0).map(() => (Math.random() - 0.5) * 0.1);

        this.modelState.layers.push({
            id: layerId,
            name: args.NAME,
            type: 'linear',
            input_dim: inputDim,
            output_dim: outputDim,
            activation: args.ACT,
            parameters: {
                weight: weightMatrix,
                bias: biasVector
            }
        });
    }

    compile_model() {
        if (this.modelState.layers.length === 0) {
            console.error('Error: No layers defined.');
            return;
        }

        const graph = { forward: [], backward: [] };
        let lastOutput = 'input_vector';

        this.modelState.layers.forEach((layer, i) => {
            const opId = `op_${i.toString().padStart(3, '0')}`;
            const actId = `act_${i.toString().padStart(3, '0')}`;
            const linearOutput = `layer_${i}_linear_out`;
            
            graph.forward.push({
                id: opId,
                type: 'linear',
                inputs: [lastOutput],
                outputs: [linearOutput],
                params: [`${layer.id}.weight`, `${layer.id}.bias`]
            });

            if (layer.activation !== 'none') {
                const actOutput = i === this.modelState.layers.length - 1 ? 'final_output' : `layer_${i}_act_out`;
                graph.forward.push({
                    id: actId,
                    type: 'activation',
                    activation_type: layer.activation,
                    inputs: [linearOutput],
                    outputs: [actOutput]
                });
                lastOutput = actOutput;
            } else {
                if (i === this.modelState.layers.length - 1) {
                    const lastOp = graph.forward[graph.forward.length - 1];
                    lastOp.outputs[0] = 'final_output';
                    lastOutput = 'final_output';
                } else {
                    lastOutput = linearOutput;
                }
            }
        });

        this.modelState.computationGraph = graph;
        this.modelState.isCompiled = true;
        
        this.rebuildParameterMap();
        console.log('Model Compiled. Graph:', graph);
    }

    // ============= 前向传播 =============

    forward_pass(args) {
        if (!this.modelState.isCompiled) return '[]';

        try {
            let input = JSON.parse(args.INPUT);
            if (!Array.isArray(input)) return '[]';
            if (Array.isArray(input[0])) input = input[0];

            this.modelState.forwardCache = { 'input_vector': input };
            
            this.modelState.computationGraph.forward.forEach(op => {
                if (op.type === 'linear') {
                    this.linearForward(op);
                } else if (op.type === 'activation') {
                    this.activationForward(op);
                }
            });

            const output = this.modelState.forwardCache['final_output'];
            return JSON.stringify(output || []);
        } catch (e) {
            console.error('Forward pass error:', e);
            return '[]';
        }
    }

    linearForward(op) {
        const weights = this.getParameter(op.params[0]);
        const bias = this.getParameter(op.params[1]);
        const input = this.modelState.forwardCache[op.inputs[0]];

        if (!weights || !bias || !input) return;

        const output = new Array(weights.length);
        for (let i = 0; i < weights.length; i++) {
            let sum = 0;
            for (let j = 0; j < input.length; j++) {
                sum += input[j] * weights[i][j];
            }
            output[i] = sum + bias[i];
        }

        this.modelState.forwardCache[op.outputs[0]] = output;
    }

    activationForward(op) {
        const input = this.modelState.forwardCache[op.inputs[0]];
        
        if (op.activation_type === 'softmax') {
            this.modelState.forwardCache[op.outputs[0]] = this.softmax(input);
        } else {
            // 调用外部激活函数扩展
            const api = window.ActivationFunctionsAPI;
            if (!api) {
                console.error('ActivationFunctionsAPI not available. Make sure activation_functions.js is loaded first.');
                this.modelState.forwardCache[op.outputs[0]] = input; // 降级处理
                return;
            }
            const output = input.map(val => api.activationFunction(val, op.activation_type));
            this.modelState.forwardCache[op.outputs[0]] = output;
        }
    }

    // ============= 反向传播 =============

    backward_pass(args) {
        if (!this.modelState.isCompiled) return;

        try {
            const target = JSON.parse(args.TARGET);
            const output = this.modelState.forwardCache['final_output'];
            
            if (!output) {
                console.error('No forward pass result found.');
                return;
            }

            let grad = this.mseLossGradient(output, target);
            
            this.modelState.gradients = {};
            
            const ops = [...this.modelState.computationGraph.forward].reverse();
            
            for (const op of ops) {
                if (op.type === 'activation') {
                    grad = this.activationBackward(grad, op);
                } else if (op.type === 'linear') {
                    grad = this.linearBackward(grad, op);
                }
            }
            
            console.log('Backward pass complete.');
        } catch (e) {
            console.error('Backward pass error:', e);
        }
    }

    activationBackward(grad, op) {
        const input = this.modelState.forwardCache[op.inputs[0]];
        const output = this.modelState.forwardCache[op.outputs[0]];
        const type = op.activation_type;

        if (type === 'softmax') {
            const sumGradY = grad.reduce((acc, g, idx) => acc + g * output[idx], 0);
            const gradInput = output.map((y, i) => y * (grad[i] - sumGradY));
            this.modelState.gradients[op.id] = gradInput;
            return gradInput;
        } else {
            // 调用外部激活函数扩展
            const api = window.ActivationFunctionsAPI;
            if (!api) {
                console.error('ActivationFunctionsAPI not available. Make sure activation_functions.js is loaded first.');
                return grad; // 降级处理
            }
            const gradInput = input.map((x, i) => 
                grad[i] * api.activationDerivative(x, type, output[i])
            );
            this.modelState.gradients[op.id] = gradInput;
            return gradInput;
        }
    }

    linearBackward(grad, op) {
        const weights = this.getParameter(op.params[0]);
        const input = this.modelState.forwardCache[op.inputs[0]];
        
        if (!weights || !input) return grad;

        const weightGrad = weights.map((row, i) => 
            row.map((_, j) => grad[i] * input[j])
        );
        
        const biasGrad = [...grad];
        
        const wKey = op.params[0];
        const bKey = op.params[1];
        
        this.modelState.gradients[wKey] = weightGrad;
        this.modelState.gradients[bKey] = biasGrad;
        
        const inputGrad = new Array(input.length).fill(0);
        for (let j = 0; j < input.length; j++) {
            for (let i = 0; i < weights.length; i++) {
                inputGrad[j] += grad[i] * weights[i][j];
            }
        }
        
        return inputGrad;
    }

    // ============= 优化器 (SGD) =============

    apply_gradients(args) {
        if (!this.modelState.isCompiled) return;

        const lr = parseFloat(args.LR);
        
        this.modelState.layers.forEach(layer => {
            const weightKey = `${layer.id}.weight`;
            const biasKey = `${layer.id}.bias`;
            
            const weightGrad = this.modelState.gradients[weightKey];
            const biasGrad = this.modelState.gradients[biasKey];
            
            if (weightGrad) {
                layer.parameters.weight = layer.parameters.weight.map((row, i) => 
                    row.map((val, j) => val - lr * weightGrad[i][j])
                );
            }
            
            if (biasGrad) {
                layer.parameters.bias = layer.parameters.bias.map((val, i) => 
                    val - lr * biasGrad[i]
                );
            }
        });
        
        this.rebuildParameterMap();
    }

    // ============= 核心算法实现 =============

    // Softmax 保留在主扩展中（向量操作）
    softmax(arr) {
        const max = Math.max(...arr);
        const exps = arr.map(x => Math.exp(x - max));
        const sum = exps.reduce((a, b) => a + b, 0);
        return exps.map(x => x / sum);
    }

    mseLossGradient(output, target) {
        const n = output.length;
        return output.map((y, i) => 2 * (y - (target[i] || 0)) / n);
    }

    // ============= 权重与参数管理 =============

    rebuildParameterMap() {
        this.modelState.parameterMap = {};
        this.modelState.layers.forEach(layer => {
            this.modelState.parameterMap[`${layer.id}.weight`] = layer.parameters.weight;
            this.modelState.parameterMap[`${layer.id}.bias`] = layer.parameters.bias;
        });
    }

    getParameter(key) {
        return this.modelState.parameterMap[key];
    }

    initializeWeight(rows, cols) {
        const scale = Math.sqrt(2 / (rows + cols));
        return Array.from({ length: rows }, () => 
            Array.from({ length: cols }, () => (Math.random() - 0.5) * 2 * scale)
        );
    }

    // ============= JSON 处理 =============

    load_weights_json(args) {
        try {
            const data = JSON.parse(args.JSON);
            
            if (data.format !== 'turbowarp-nn-weights') {
                console.error('Invalid weight format');
                return;
            }
            
            this.modelState.layers = data.layers.map(layer => {
                let w = layer.parameters.weight;
                let b = layer.parameters.bias;
                
                if (w && w.data) w = w.data;
                if (b && b.data) b = b.data;

                return {
                    id: layer.id,
                    name: layer.name || layer.id,
                    type: layer.type,
                    input_dim: layer.input_dim,
                    output_dim: layer.output_dim,
                    activation: layer.activation || 'none',
                    parameters: {
                        weight: w,
                        bias: b
                    }
                };
            });
            
            this.modelState.computationGraph = data.computation_graph;
            this.modelState.isCompiled = true;
            
            this.rebuildParameterMap();
            console.log('Weights loaded. Layers:', this.modelState.layers.length);
        } catch (e) {
            console.error('JSON Parse Error:', e);
        }
    }

    export_weights() {
        const exportLayers = this.modelState.layers.map(layer => ({
            id: layer.id,
            type: layer.type,
            input_dim: layer.input_dim,
            output_dim: layer.output_dim,
            activation: layer.activation,
            parameters: {
                weight: {
                    shape: [layer.output_dim, layer.input_dim],
                    data: layer.parameters.weight
                },
                bias: {
                    shape: [layer.output_dim],
                    data: layer.parameters.bias
                }
            }
        }));

        const exportData = {
            format: 'turbowarp-nn-weights',
            version: '1.1',
            model_meta: {
                name: 'Exported Model',
                total_layers: this.modelState.layers.length,
                created: new Date().toISOString()
            },
            layers: exportLayers,
            computation_graph: this.modelState.computationGraph
        };
        
        return JSON.stringify(exportData, null, 2);
    }

    // ============= 工具函数 =============

    get_output() {
        const output = this.modelState.forwardCache['final_output'];
        return JSON.stringify(output || []);
    }

    matrix_multiply(args) {
        try {
            const a = JSON.parse(args.A);
            const b = JSON.parse(args.B);
            
            if (!a.length || !b.length || a[0].length !== b.length) return '[]';
            
            const result = Array(a.length).fill(0).map(() => Array(b[0].length).fill(0));
            
            for (let i = 0; i < a.length; i++) {
                for (let j = 0; j < b[0].length; j++) {
                    for (let k = 0; k < b.length; k++) {
                        result[i][j] += a[i][k] * b[k][j];
                    }
                }
            }
            return JSON.stringify(result);
        } catch (e) {
            return '[]';
        }
    }
}

Scratch.extensions.register(new NeuralNetworkExtension());