/// Optimal rectangular assignment (Hungarian algorithm, shortest augmenting
/// path / Jonker-Volgenant style), equivalent to SciPy's
/// `linear_sum_assignment`: minimises total cost, assigning
/// min(rows, cols) pairs.
library;

class AssignmentResult {
  final List<(int, int)> matches;
  final List<int> unmatchedRows;
  final List<int> unmatchedCols;
  const AssignmentResult(this.matches, this.unmatchedRows, this.unmatchedCols);
}

/// Returns the optimal row->col pairs for [cost] (rows x cols).
List<(int, int)> linearSumAssignment(List<List<double>> cost) {
  final n = cost.length;
  if (n == 0) return const [];
  final m = cost[0].length;
  if (m == 0) return const [];
  // Work on a matrix with rows <= cols (transpose if needed).
  final transpose = n > m;
  final rows = transpose ? m : n;
  final cols = transpose ? n : m;
  double c(int i, int j) => transpose ? cost[j][i] : cost[i][j];

  const inf = double.infinity;
  // 1-indexed potentials, as in the classic e-maxx formulation.
  final u = List<double>.filled(rows + 1, 0);
  final v = List<double>.filled(cols + 1, 0);
  final p = List<int>.filled(cols + 1, 0); // p[j] = row assigned to column j
  final way = List<int>.filled(cols + 1, 0);
  for (var i = 1; i <= rows; i++) {
    p[0] = i;
    var j0 = 0;
    final minv = List<double>.filled(cols + 1, inf);
    final used = List<bool>.filled(cols + 1, false);
    do {
      used[j0] = true;
      final i0 = p[j0];
      var delta = inf;
      var j1 = 0;
      for (var j = 1; j <= cols; j++) {
        if (used[j]) continue;
        final cur = c(i0 - 1, j - 1) - u[i0] - v[j];
        if (cur < minv[j]) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1 = j;
        }
      }
      for (var j = 0; j <= cols; j++) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] != 0);
    do {
      final j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 != 0);
  }
  final out = <(int, int)>[];
  for (var j = 1; j <= cols; j++) {
    if (p[j] != 0) {
      final r = p[j] - 1, col = j - 1;
      out.add(transpose ? (col, r) : (r, col));
    }
  }
  out.sort((a, b) => a.$1.compareTo(b.$1));
  return out;
}

/// Hungarian matching; pairs costing more than [thresh] are rejected.
AssignmentResult linearAssignment(List<List<double>> cost, int nRows, int nCols, double thresh) {
  if (nRows == 0 || nCols == 0) {
    return AssignmentResult(const [], List.generate(nRows, (i) => i), List.generate(nCols, (i) => i));
  }
  final matches = <(int, int)>[];
  final ur = {for (var i = 0; i < nRows; i++) i};
  final uc = {for (var i = 0; i < nCols; i++) i};
  for (final (r, c) in linearSumAssignment(cost)) {
    if (cost[r][c] <= thresh) {
      matches.add((r, c));
      ur.remove(r);
      uc.remove(c);
    }
  }
  return AssignmentResult(matches, ur.toList()..sort(), uc.toList()..sort());
}
