class MathTool { 
  MatrixMultiplication(a, b) {
    return a.map((row, i) =>
      b[0].map((_, j) =>
        row.reduce((sum, val, k) => sum + val * b[k][j], 0)
      )
    );
  }

  transpose(matrix) {
    const maxCols = Math.max(...matrix.map(row => row.length));
    const result = [];

    for (let i = 0; i < maxCols; i++) {
      result[i] = matrix.map(row => row[i] !== undefined ? row[i] : null);
    }
    
    return result;
  }
}


class Tensor {

}

class EtudeMLExtension {
  getInfo() {
    return {
      id: 'EtudeTurboWarpML',
      name: 'Etude-TurboWarp-ML',
      color1: '#4C97FF',
      color2: '#3d85c6',
      color3: '#2e5d8f',
      author: 'Asuka | BlueIrisSky',
      version: '0.1.0',

    }
  }
}
Scratch.extensions.register(new EtudeMLExtension());