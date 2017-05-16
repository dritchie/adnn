'use strict';

var Tensor = require('../../tensor.js');
var graph = require('../graph.js');
var Node = graph.Node;
var func = require('../func.js');
//var fns = require('../functions.js').fns;
// Linear Algebra  -----------------------------------------------------

var fns = {tensor: {}};

fns.tensor.transpose = func.newUnaryFunction({
  OutputType: Tensor,
  name: 'transpose',
  forward: function(a) {
    return a.transpose();
  },
  backward: function(a) {
    var h = this.x.dims[0];
    var w = this.x.dims[1];
    for (var i = 0; i < h; i++) {
      for (var j = 0; j < w; j++) {
        a.dx.data[j * h + i] += this.dx.data[i * w + j];
      }
    }
  }
});

fns.tensor.diag = func.newUnaryFunction({
  OutputType: Tensor,
  name: 'diagonal',
  forward: function(a) {
    return a.diag();
  },
  backward: function(a) {
    var n = a.dx.dims[0];
    for (var i = 0; i < n; i++) {
      a.dx.data[i * (n + 1)] += this.dx.data[i * (n + 1)];
    }
  }
});

fns.tensor.diagonal = func.newUnaryFunction({
  OutputType: Tensor,
  name: 'diagonal',
  forward: function(a) {
    return a.diagonal();
  },
  backward: function(a) {
    var n = a.dx.dims[0];
    for (var i = 0; i < n; i++) {
      a.dx.data[i] += this.dx.data[i * (n + 1)];
    }
  }
});

fns.tensor.inverse = func.newUnaryFunction({
  OutputType: Tensor,
  name: 'inverse',
  forward: function(A) {
    return A.inverse();
  },
  backward: function(A) {
    var xT = this.x.transpose();
    A.dx = A.dx.add(xT.dot(this.dx).dot(xT).neg());
  }
});

fns.tensor.determinant = func.newUnaryFunction({
  OutputType: Number,
  name: 'determinant',
  forward: function(A) {
    return A.determinant();
  },
  backward: function(A) {
    // A is square matrix.
    // Assume A is invertable.
    var n = A.x.dims[0];
    var invA = A.x.inverse();
    for (var i = 0; i < n; i++) {
      for (var j = 0; j < n; j++) {
        A.dx.data[i * n + j] += this.x * this.dx * invA.data[j * n + i];
      }
    }
  }
});

fns.tensor.dot = func.newBinaryFunction({
  OutputType: Tensor,
  name: 'dot',
  forward: function(a, b) {
    return a.dot(b);
  },
  backward1: function(A, B) {
    var Ap = A instanceof Node ? A.x : A;
    var Bp = B instanceof Node ? B.x : B;

    var Ah = Ap.dims[0];
    var Aw = Ap.dims[1];
    var Bw = Bp.dims[1];
    var wout = Bw;

    for (var l = 0; l < Ah; l++) {
      for (var m = 0; m < Aw; m++) {
        var z = 0;
        for (var j = 0; j < wout; j++) {
          z += this.dx.data[l * wout + j] * Bp.data[m * Bw + j];
        }
        A.dx.data[l * Aw + m] += z;
      }
    }
  },
  backward2: function(A, B) {
    var Ap = A instanceof Node ? A.x : A;
    var Bp = B instanceof Node ? B.x : B;

    var Ah = Ap.dims[0];
    var Aw = Ap.dims[1];
    var Bh = Bp.dims[0];
    var Bw = Bp.dims[1];
    var wout = Bw;

    for (var l = 0; l < Bh; l++) {
      for (var m = 0; m < Bw; m++) {
        var z = 0;
        for (var i = 0; i < Ah; i++) {
          z += this.dx.data[i * wout + m] * Ap.data[i * Aw + l];
        }
        B.dx.data[l * Bw + m] += z;
      }
    }

  }
});


module.exports = fns;
