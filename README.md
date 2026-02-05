# 差分隐私会计工具

这是一个基于Opacus的差分隐私会计工具，用于计算机器学习训练中的隐私预算。

## 快速开始

### 1. 安装依赖
```bash
pip install torch torchvision opacus numpy matplotlib
```
### 2. 基本使用
```python
from dp_accountant_enhanced import EnhancedDPAccountant

# 初始化会计器
accountant = EnhancedDPAccountant(dataset_size=50000)

# 计算隐私预算
epsilon, delta, alpha = accountant.compute_epsilon(
    batch_size=256,
    steps=1000,
    noise_multiplier=1.0
)
print(f"ε={epsilon:.2f}, δ={delta:.2e}")

# 反推噪声参数
noise, actual_eps = accountant.compute_noise_for_target_epsilon(
    batch_size=256,
    steps=1000,
    target_epsilon=2.0
)
print(f"需要噪声乘子: σ={noise:.3f}")
```
文件说明
privacy_accountant_opacus.py: 第4天的代码，基础会计工具

dp_accountant_enhanced.py: 第5天的代码，增强版会计工具

作者
[你的名字]

日期
2024年1月

