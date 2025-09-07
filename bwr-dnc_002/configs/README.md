# Configuration Examples

This directory contains example configuration files for different use cases of the BWR-DNC model.

## Available Configurations

### minimal.yaml
Minimal configuration for testing and development:
```yaml
model:
  d_model: 256
  n_layers: 4
  n_heads: 8
  max_seq_len: 512
  dropout: 0.1

memory:
  memory_slots: [256, 128, 64]
  memory_integration_layers: [2, 3]

training:
  learning_rate: 0.001
  batch_size: 8
  num_epochs: 10
```

### standard.yaml
Standard configuration for most use cases:
```yaml
model:
  d_model: 512
  n_layers: 6
  n_heads: 8
  max_seq_len: 2048
  dropout: 0.1

memory:
  memory_slots: [2048, 1024, 512]
  memory_integration_layers: [4, 5]

training:
  learning_rate: 0.0003
  batch_size: 16
  num_epochs: 50
  patience: 5
```

### large.yaml
Large configuration for high-performance applications:
```yaml
model:
  d_model: 1024
  n_layers: 12
  n_heads: 16
  max_seq_len: 4096
  dropout: 0.1

memory:
  memory_slots: [4096, 2048, 1024, 512]
  memory_integration_layers: [8, 9, 10, 11]

training:
  learning_rate: 0.0001
  batch_size: 32
  num_epochs: 100
  patience: 10
```
