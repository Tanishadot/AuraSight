/// Constant-velocity Kalman filter on (cx, cy, aspect, height), as used by
/// SORT / DeepSORT / ByteTrack. Port of `KalmanFilterXYAH` in
/// backend/tracking/bytetrack.py.
library;

import 'dart:math' as math;

typedef Vec = List<double>;
typedef Mat = List<List<double>>;

Mat _zeros(int r, int c) => List.generate(r, (_) => List<double>.filled(c, 0));

Mat _diagSq(List<double> std) {
  final m = _zeros(std.length, std.length);
  for (var i = 0; i < std.length; i++) {
    m[i][i] = std[i] * std[i];
  }
  return m;
}

class KalmanFilterXYAH {
  static const _ndim = 4;
  static const _stdPos = 1.0 / 20;
  static const _stdVel = 1.0 / 160;

  (Vec, Mat) initiate(Vec measurement) {
    final mean = [...measurement, 0.0, 0.0, 0.0, 0.0];
    final h = measurement[3];
    final std = [
      2 * _stdPos * h, 2 * _stdPos * h, 1e-2, 2 * _stdPos * h, //
      10 * _stdVel * h, 10 * _stdVel * h, 1e-5, 10 * _stdVel * h,
    ];
    return (mean, _diagSq(std));
  }

  (Vec, Mat) predict(Vec mean, Mat cov) {
    final h = mean[3];
    final q = _diagSq([
      _stdPos * h, _stdPos * h, 1e-2, _stdPos * h, //
      _stdVel * h, _stdVel * h, 1e-5, _stdVel * h,
    ]);
    // F = [[I, I], [0, I]]
    final m = List<double>.of(mean);
    for (var i = 0; i < _ndim; i++) {
      m[i] = mean[i] + mean[i + _ndim];
    }
    // F P F^T
    final fp = _zeros(8, 8);
    for (var i = 0; i < 8; i++) {
      for (var j = 0; j < 8; j++) {
        fp[i][j] = cov[i][j] + (i < _ndim ? cov[i + _ndim][j] : 0);
      }
    }
    final out = _zeros(8, 8);
    for (var i = 0; i < 8; i++) {
      for (var j = 0; j < 8; j++) {
        out[i][j] = fp[i][j] + (j < _ndim ? fp[i][j + _ndim] : 0) + q[i][j];
      }
    }
    return (m, out);
  }

  (Vec, Mat) update(Vec mean, Mat cov, Vec measurement) {
    final h = mean[3];
    final std = [_stdPos * h, _stdPos * h, 1e-1, _stdPos * h];
    // S = H P H^T + R  (H selects the first four components)
    final s = _zeros(4, 4);
    for (var i = 0; i < 4; i++) {
      for (var j = 0; j < 4; j++) {
        s[i][j] = cov[i][j] + (i == j ? std[i] * std[i] : 0);
      }
    }
    // K^T = solve(S, (P H^T)^T)  -> rhs is 4 x 8: rhs[i][k] = P[k][i]
    final rhs = _zeros(4, 8);
    for (var i = 0; i < 4; i++) {
      for (var k = 0; k < 8; k++) {
        rhs[i][k] = cov[k][i];
      }
    }
    final kt = _solve(s, rhs); // 4 x 8
    final innov = [for (var i = 0; i < 4; i++) measurement[i] - mean[i]];
    final newMean = List<double>.of(mean);
    for (var k = 0; k < 8; k++) {
      var acc = 0.0;
      for (var i = 0; i < 4; i++) {
        acc += innov[i] * kt[i][k];
      }
      newMean[k] += acc;
    }
    // P' = P - K S K^T
    final ks = _zeros(8, 4); // K S
    for (var a = 0; a < 8; a++) {
      for (var j = 0; j < 4; j++) {
        var acc = 0.0;
        for (var i = 0; i < 4; i++) {
          acc += kt[i][a] * s[i][j];
        }
        ks[a][j] = acc;
      }
    }
    final newCov = _zeros(8, 8);
    for (var a = 0; a < 8; a++) {
      for (var b = 0; b < 8; b++) {
        var acc = 0.0;
        for (var j = 0; j < 4; j++) {
          acc += ks[a][j] * kt[j][b];
        }
        newCov[a][b] = cov[a][b] - acc;
      }
    }
    return (newMean, newCov);
  }
}

/// Solves A X = B for X (A n x n, B n x m) by Gaussian elimination with
/// partial pivoting (as LAPACK's gesv does).
Mat _solve(Mat a0, Mat b0) {
  final n = a0.length, m = b0[0].length;
  final a = [for (final r in a0) List<double>.of(r)];
  final b = [for (final r in b0) List<double>.of(r)];
  for (var col = 0; col < n; col++) {
    var piv = col;
    for (var r = col + 1; r < n; r++) {
      if (a[r][col].abs() > a[piv][col].abs()) piv = r;
    }
    if (piv != col) {
      final ta = a[col];
      a[col] = a[piv];
      a[piv] = ta;
      final tb = b[col];
      b[col] = b[piv];
      b[piv] = tb;
    }
    final d = a[col][col];
    for (var r = col + 1; r < n; r++) {
      final f = a[r][col] / d;
      if (f == 0) continue;
      for (var c = col; c < n; c++) {
        a[r][c] -= f * a[col][c];
      }
      for (var c = 0; c < m; c++) {
        b[r][c] -= f * b[col][c];
      }
    }
  }
  final x = _zeros(n, m);
  for (var r = n - 1; r >= 0; r--) {
    for (var c = 0; c < m; c++) {
      var acc = b[r][c];
      for (var k = r + 1; k < n; k++) {
        acc -= a[r][k] * x[k][c];
      }
      x[r][c] = acc / a[r][r];
    }
  }
  return x;
}

/// (x1, y1, x2, y2) -> (cx, cy, aspect = w / h, h)
Vec xyxyToXyah(List<double> b) {
  final w = b[2] - b[0], h = b[3] - b[1];
  return [b[0] + w / 2, b[1] + h / 2, w / math.max(h, 1e-6), h];
}
