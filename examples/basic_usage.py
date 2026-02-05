"""
差分隐私会计工具 - 基础使用示例
"""

import sys
import os

# 添加父目录到Python路径，这样我们可以导入我们的工具
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dp_accountant_enhanced import EnhancedDPAccountant
    print("✓ 成功导入EnhancedDPAccountant")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    print("请确保dp_accountant_enhanced.py在项目根目录")
    sys.exit(1)

def main():
    print("=" * 60)
    print("差分隐私会计工具 - 基础使用示例")
    print("=" * 60)
    
    # 1. 初始化会计器
    print("\n1. 初始化会计器")
    accountant = EnhancedDPAccountant(dataset_size=50000)
    print(f"数据集大小: {accountant.dataset_size}")
    
    # 2. 计算隐私预算
    print("\n2. 计算隐私预算")
    print("参数: batch_size=256, steps=1000, noise_multiplier=1.0")
    
    epsilon, delta, alpha = accountant.compute_epsilon(
        batch_size=256,
        steps=1000,
        noise_multiplier=1.0
    )
    
    print(f"计算结果:")
    print(f"  ε (隐私预算) = {epsilon:.2f}")
    print(f"  δ            = {delta:.2e}")
    print(f"  最优α        = {alpha:.1f}")
    
    # 3. 反推噪声参数
    print("\n3. 反推噪声参数")
    print("目标: ε=2.0, batch_size=256, steps=1000")
    
    noise, actual_eps = accountant.compute_noise_for_target_epsilon(
        batch_size=256,
        steps=1000,
        target_epsilon=2.0
    )
    
    print(f"反推结果:")
    print(f"  需要噪声乘子 σ = {noise:.3f}")
    print(f"  实际ε值        = {actual_eps:.3f}")
    
    # 4. 生成ε网格
    print("\n4. 生成ε网格配置")
    configs = accountant.generate_epsilon_grid(
        target_epsilons=[0.5, 1.0, 2.0, 4.0, 8.0],
        batch_size=256,
        steps=10000
    )
    
    print(f"生成了 {len(configs)} 个配置")
    
    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()