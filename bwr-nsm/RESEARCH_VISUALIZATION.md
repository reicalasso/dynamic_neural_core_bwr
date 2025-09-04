# Research Visualization Layer (v0.1)

This document describes the initial research telemetry layer added to the project.

## Endpoints

REST: `GET /research/metrics` – returns the latest snapshot (JSON)
WebSocket: `ws://localhost:8000/ws/research` – pushes `{ type: "research_metrics", data: <payload> }` every ~2s

## Metrics Schema
```
timestamp: ISO8601
memory: { levels: [{ level, slots, active_slots, avg_salience, avg_age, avg_access }], total_slots, total_active_slots }
generation: { total_requests, total_generated_tokens, avg_tokens_per_request, recent_tps }
performance: { gpu_memory_mb|null, gpu_utilization|null }
compression: { levels, hierarchical_factor|null }
```

## Frontend
Page: `/research` (Next.js) – Client-only dynamic import
Component: `ResearchDashboard` with fallback polling if WS not available

## Design Principles
1. Zero extra frontend deps (pure Tailwind + React) for fast iteration
2. Always return stable field names (use `null` or `0` instead of omitting)
3. Lightweight backend (no blocking, async broadcast loop)
4. Degrade gracefully: WebSocket → Poll → Static

## Future Extensions (Suggested)
* Salience distribution histogram (bucketization)
* Attention snapshots (sampled routes)
* Export: download NDJSON / CSV
* Session comparison overlay
* Latency breakdown (forward vs. memory read vs. compression)

---
Implementation version: 0.1 (baseline) – safe to extend.
