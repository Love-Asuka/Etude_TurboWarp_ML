class main {
  getInfo() {
    return {
      id: 'EtudeTurboWarpcore',
      name: 'Etude-TurboWarp-ML',
      author: 'Asuka | Lin Xin',
      version: '1.0.1',
      blocks: [
        {
          opcode: 'Matrix_multiplication',
          blockType: Scratch.BlockType.REPORTER,
          text: 'Matrix [A] × [B]',
          arguments: {
            A: {
              type: Scratch.ArgumentType.STRING,
              defaultValue: ''
            },
            B: {
              type: Scratch.ArgumentType.STRING,
              defaultValue: ''
            }
          }
        }
        ,
        {
          opcode: 'Matrix_addition',
          blockType: Scratch.BlockType.REPORTER,
          text: 'Matrix [A] + [B]',
          arguments: {
            A: {
              type: Scratch.ArgumentType.STRING,
              defaultValue: ''
            },
            B: {
              type: Scratch.ArgumentType.STRING,
              defaultValue: ''
            }
          }
        }
      ]
    };
  }
  
  
  
  matrix_multiplication(args) {
    const a = args.A;
    const b = args.B;

    if (!Array.isArray(a) || !Array.isArray(b) || a.length === 0 || b.length === 0) {
        return [];
    }
    if (!a[0] || a[0].length !== b.length) {
        return []; 
    }
    let result = [];
    for (let i = 0; i < a.length; i++) {
        result[i] = [];
        for (let j = 0; j < b[0].length; j++) {
            let sum = 0;
            for (let k = 0; k < a[0].length; k++) {
                sum += a[i][k] * b[k][j];
            }
            result[i][j] = sum;
        }
    }
    return result; 
  }

  matrix_add(args) {
    const a = args.A;
    const b = args.B;

    if (!Array.isArray(a) || !Array.isArray(b) || a.length === 0 || b.length === 0) {
        return [];
    }
    if (a.length !== b.length || a[0].length !== b[0].length) {
        return []; 
    }
    let result = [];
    for (let i = 0; i < a.length; i++) {
        result[i] = [];
        for (let j = 0; j < a[0].length; j++) {
            result[i][j] = a[i][j] + b[i][j];
        }
    }
    return result; 
  }

}
Scratch.extensions.register(new main());