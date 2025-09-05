"""Research Metrics Collection
============================

Light‑weight aggregator that extracts structured telemetry from the DNC /
AdvancedStateBank for the research visualization layer.  It is intentionally
dependency‑free (no numpy/pandas) and returns JSON‑serialisable primitives.

Metrics Schema (v0.1):
{
  "timestamp": iso8601 str,
  "memory": {
      "levels": [
          {
            "level": int,
            "slots": int,
            "active_slots": int,              # salience > threshold
            "avg_salience": float,
            "avg_age": float,
            "avg_access": float
          }, ...
      ],
      "total_slots": int,
      "total_active_slots": int
  },
  "generation": {
      "total_requests": int,
      "total_generated_tokens": int,
      "avg_tokens_per_request": float,
      "recent_tps": float                      # tokens/sec (last window)
  },
  "performance": {                             # Placeholders for future GPU stats
      "gpu_memory_mb": float | null,
      "gpu_utilization": float | null
  },
  "compression": {
      "levels": int,
      "hierarchical_factor": float | null
  }
}

The class is defensive: if a metric cannot be derived it falls back to None
or zero so the frontend never breaks due to missing fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import torch


@dataclass
class GenerationCounters:
    total_requests: int = 0
    total_generated_tokens: int = 0
    # rolling window stats are computed externally and injected
    recent_tokens_per_sec: float = 0.0


class ResearchMetricsAggregator:
    def __init__(self, model_ref_getter, gen_counters: GenerationCounters, salience_threshold: float = 0.1):
        """
        model_ref_getter: callable returning the current model (or None)
        gen_counters: shared counters updated by generation route
        salience_threshold: salience value above which a slot is considered active
        """
        self._get_model = model_ref_getter
        self.gen = gen_counters
        self.salience_threshold = salience_threshold

    def _memory_metrics(self, model) -> Dict[str, Any]:
        levels_info = []
        total_slots = 0
        total_active = 0

        state = getattr(model, 'state', None)
        if not state or not hasattr(state, 'levels'):
            return {
                "levels": [],
                "total_slots": 0,
                "total_active_slots": 0
            }

        for idx, level in enumerate(state.levels):
            try:
                K = level['K']
                sal = level['salience']
                age = level.get('age')
                access = level.get('access_count')
                slots = K.shape[0]
                active_slots = int((sal > self.salience_threshold).sum().item())
                levels_info.append({
                    "level": idx,
                    "slots": slots,
                    "active_slots": active_slots,
                    "avg_salience": float(sal.mean().item()),
                    "avg_age": float(age.mean().item()) if age is not None else 0.0,
                    "avg_access": float(access.mean().item()) if access is not None else 0.0
                })
                total_slots += slots
                total_active += active_slots
            except Exception:
                continue

        return {
            "levels": levels_info,
            "total_slots": total_slots,
            "total_active_slots": total_active
        }

    def _compression_metrics(self, model) -> Dict[str, Any]:
        state = getattr(model, 'state', None)
        if not state or not hasattr(state, 'levels'):
            return {"levels": 0, "hierarchical_factor": None}
        try:
            level_sizes = [lvl['K'].shape[0] for lvl in state.levels]
            if len(level_sizes) >= 2:
                # approximate geometric compression factor
                factor = level_sizes[0] / level_sizes[-1] if level_sizes[-1] > 0 else None
            else:
                factor = None
            return {"levels": len(level_sizes), "hierarchical_factor": factor}
        except Exception:
            return {"levels": 0, "hierarchical_factor": None}

    def _performance_metrics(self) -> Dict[str, Any]:
        # Placeholder: real implementation would integrate torch.cuda + perf monitor
        if torch.cuda.is_available():
            try:
                mem = torch.cuda.memory_allocated() / (1024 * 1024)
                # util would need pynvml or similar; omitted for portability
                return {"gpu_memory_mb": round(mem, 2), "gpu_utilization": None}
            except Exception:
                pass
        return {"gpu_memory_mb": None, "gpu_utilization": None}

    def collect(self) -> Dict[str, Any]:
        model = self._get_model()
        memory = self._memory_metrics(model) if model else {"levels": [], "total_slots": 0, "total_active_slots": 0}
        compression = self._compression_metrics(model) if model else {"levels": 0, "hierarchical_factor": None}
        performance = self._performance_metrics()

        gen = self.gen
        avg_tokens_per_req = (gen.total_generated_tokens / gen.total_requests) if gen.total_requests else 0.0

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "memory": memory,
            "compression": compression,
            "performance": performance,
            "generation": {
                "total_requests": gen.total_requests,
                "total_generated_tokens": gen.total_generated_tokens,
                "avg_tokens_per_request": avg_tokens_per_req,
                "recent_tps": gen.recent_tokens_per_sec
            }
        }
