# 🚀Self-pruning-neural-n-w
Developed a self-pruning neural network using PyTorch that removes redundant neurons during training to reduce model size and computation. Implemented magnitude-based pruning to maintain accuracy while improving efficiency and faster inference.

## 🔑 Key Idea
Each weight is associated with a learnable gate:
```math
w_{eff} = w \cdot \sigma(\tau \cdot g)

- g → learnable gate score

- τ → temperature scaling (sharpens pruning)
 
- Gate ≈ 0 → weight removed

- Gate ≈ 1 → weight kept  
