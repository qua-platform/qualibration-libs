from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Sequence, Optional


@dataclass(frozen=True)
class QubitGrid:
    coords: Dict[str, Tuple[int, int]]
    shape: Optional[Tuple[int, int]] = None

    def resolve(self, present_qubits: Sequence[str]) -> Tuple[int, int, Dict[str, Tuple[int, int]]]:
        positions: Dict[str, Tuple[int, int]] = {}
        rows_max = 0
        cols_max = 0
        for q in present_qubits:
            if q not in self.coords:
                continue
            r, c = self.coords[q]
            r_idx, c_idx = r + 1, c + 1
            positions[q] = (r_idx, c_idx)
            rows_max = max(rows_max, r_idx)
            cols_max = max(cols_max, c_idx)

        if self.shape is not None:
            n_rows, n_cols = self.shape
        else:
            n_rows, n_cols = rows_max, cols_max

        return n_rows, n_cols, positions