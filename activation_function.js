class ActivationFunctionsExtension {
    constructor() {
        // 将API暴露到全局，供主神经网络扩展调用
        window.ActivationFunctionsAPI = {
            activationFunction: this.activationFunction.bind(this),
            activationDerivative: this.activationDerivative.bind(this)
        };
    }
    
    getInfo() {
        return {
            id: 'ActivationFunctions',
            name: '激活函数工具',
            author: 'Asuka | Lin Xin',
            version: '1.0.0',
            blocks: [
                {
                    opcode: 'calc_activation',
                    blockType: Scratch.BlockType.REPORTER,
                    text: '计算 [TYPE] 激活值 x=[X]',
                    arguments: {
                        TYPE: { type: Scratch.ArgumentType.STRING, menu: 'activationMenu' },
                        X: { type: Scratch.ArgumentType.NUMBER, defaultValue: 0.5 }
                    }
                },
                {
                    opcode: 'calc_derivative',
                    blockType: Scratch.BlockType.REPORTER,
                    text: '计算 [TYPE] 导数 x=[X]',
                    arguments: {
                        TYPE: { type: Scratch.ArgumentType.STRING, menu: 'activationMenu' },
                        X: { type: Scratch.ArgumentType.NUMBER, defaultValue: 0.5 }
                    }
                }
            ],
            menus: {
                activationMenu: ['relu', 'tanh', 'sigmoid', 'none']
            }
        };
    }
    
    calc_activation(args) {
        const x = parseFloat(args.X);
        const result = this.activationFunction(x, args.TYPE);
        return isNaN(result) ? 0 : result;
    }
    
    calc_derivative(args) {
        const x = parseFloat(args.X);
        const result = this.activationDerivative(x, args.TYPE);
        return isNaN(result) ? 0 : result;
    }
    
    // 核心函数 - 供其他扩展调用
    activationFunction(x, type) {
        switch (type) {
            case 'relu': return Math.max(0, x);
            case 'tanh': return Math.tanh(x);
            case 'sigmoid': return 1 / (1 + Math.exp(-x));
            default: return x;
        }
    }
    
    activationDerivative(x, type, activatedValue = null) {
        switch (type) {
            case 'relu': return x > 0 ? 1 : 0;
            case 'tanh': {
                const t = activatedValue !== null ? activatedValue : Math.tanh(x);
                return 1 - t * t;
            }
            case 'sigmoid': {
                const s = activatedValue !== null ? activatedValue : (1 / (1 + Math.exp(-x)));
                return s * (1 - s);
            }
            default: return 1;
        }
    }
}

Scratch.extensions.register(new ActivationFunctionsExtension());