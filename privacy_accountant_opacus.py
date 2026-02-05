import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from typing import List

# 配置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
import json
import numpy as np
from datetime import datetime

# Opacus 1.5.4 导入方式
import opacus.accountants.rdp as rdp_module
privacy_analysis = rdp_module.privacy_analysis
compute_rdp = privacy_analysis.compute_rdp
get_privacy_spent = privacy_analysis.get_privacy_spent

class DPAccountant:
    def __init__(self, dataset_size, delta_method='1/N'):
        """
        初始化隐私会计器
        
        Args:
            dataset_size: 数据集大小
            delta_method: δ计算方法，可选 '1/N', '1/N^1.1', 'fixed'
        """
        self.dataset_size = dataset_size
        self.delta_method = delta_method
        
        # RDP orders (λ值) - 参考Opacus官方推荐
        self.orders = list(np.linspace(1.1, 10, 90)) + list(range(11, 65))
        
        # 训练记录
        self.history = []
        
    def compute_delta(self):
        """计算δ值"""
        if self.delta_method == '1/N':
            return 1.0 / self.dataset_size
        elif self.delta_method == '1/N^1.1':
            return 1.0 / (self.dataset_size ** 1.1)
        elif self.delta_method.startswith('fixed:'):
            return float(self.delta_method.split(':')[1])
        else:
            # 尝试直接转换为float
            try:
                return float(self.delta_method)
            except:
                raise ValueError(f"不支持的delta_method: {self.delta_method}")
    
    def compute_epsilon(self, batch_size, steps, noise_multiplier, clip_norm=1.0):
        """计算隐私预算ε"""
        
        # 计算δ
        delta = self.compute_delta()
        
        # 采样率
        sampling_rate = batch_size / self.dataset_size
        
        # 计算RDP
        rdp = compute_rdp(
            q=sampling_rate,
            noise_multiplier=noise_multiplier,
            steps=steps,
            orders=self.orders
        )
        
        # 转换为(ε, δ) - 使用正确的参数名：delta
        eps, optimal_alpha = get_privacy_spent(
            orders=self.orders,
            rdp=rdp,
            delta=delta  # 关键修改：使用delta而不是target_delta
        )
        
        # 记录
        record = {
            'timestamp': datetime.now().isoformat(),
            'batch_size': batch_size,
            'steps': steps,
            'noise_multiplier': noise_multiplier,
            'clip_norm': clip_norm,
            'epsilon': eps,
            'delta': delta,
            'optimal_alpha': optimal_alpha,
            'sampling_rate': sampling_rate
        }
        self.history.append(record)
        
        return eps, delta, optimal_alpha
    
    def compute_noise_for_target_epsilon(self, batch_size, steps, target_epsilon, 
                                         delta=None, noise_range=(0.1, 10.0), tolerance=0.01):
        """给定目标ε，反推噪声乘子σ（二分查找）"""
        if delta is None:
            delta = self.compute_delta()
            
        low, high = noise_range
        best_noise = None
        
        for _ in range(30):  # 二分查找最多30次
            mid = (low + high) / 2
            eps, _, _ = self.compute_epsilon(batch_size, steps, mid)
            
            # 记录最接近的值
            if best_noise is None or abs(eps - target_epsilon) < abs(best_noise[0] - target_epsilon):
                best_noise = (eps, mid)
            
            if abs(eps - target_epsilon) < tolerance:
                return mid, eps
            
            if eps < target_epsilon:
                high = mid  # ε太小 → 需要减小噪声（σ）
            else:
                low = mid   # ε太大 → 需要增大噪声（σ）
                
        # 返回最接近的值
        return best_noise[1], best_noise[0]
    
    def save_config(self, filepath):
        """保存配置"""
        config = {
            'dataset_size': self.dataset_size,
            'delta_method': self.delta_method,
            'history': self.history
        }
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    
    def plot_privacy_curve(self, save_path='privacy_curve.png'):
        """绘制隐私曲线"""
        import matplotlib.pyplot as plt
        
        if not self.history:
            print("没有历史记录可绘制")
            return
            
        steps = [r['steps'] for r in self.history]
        epsilons = [r['epsilon'] for r in self.history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, epsilons, 'b-o', linewidth=2, markersize=6)
        plt.xlabel('训练步数', fontsize=12)
        plt.ylabel('ε (隐私预算)', fontsize=12)
        plt.title('隐私消耗曲线 (Opacus 1.5.4)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"图表已保存到 {save_path}")

# 测试
if __name__ == "__main__":
    # 创建会计器
    accountant = DPAccountant(dataset_size=50000, delta_method='1/N')
    
    print("=" * 60)
    print("隐私会计工具测试 (Opacus 1.5.4)")
    print("=" * 60)
    print(f"数据集大小: 50000")
    print(f"δ计算方法: {accountant.delta_method}")
    print(f"δ值: {accountant.compute_delta():.2e}")
    
    print("\n" + "=" * 60)
    print("测试正向计算: 给定参数计算ε")
    print("=" * 60)
    
    # 模拟训练过程
    test_cases = [
        (100, 256, 1.0),
        (500, 256, 1.0),
        (1000, 256, 1.0),
        (2000, 256, 1.0),
        (5000, 256, 1.0)
    ]
    
    print(f"{'Steps':>8} {'Batch Size':>10} {'Noise (σ)':>10} {'ε':>10} {'δ':>12} {'α':>8}")
    print("-" * 68)
    
    for steps, batch_size, noise in test_cases:
        eps, delta, alpha = accountant.compute_epsilon(batch_size, steps, noise)
        print(f"{steps:8d} {batch_size:10d} {noise:10.1f} {eps:10.2f} {delta:12.2e} {alpha:8.1f}")
    
    print("\n" + "=" * 60)
    print("测试反向计算: 给定目标ε计算所需噪声")
    print("=" * 60)
    
    # 给定目标ε，计算需要的噪声
    target_epsilon = 2.0
    noise, actual_epsilon = accountant.compute_noise_for_target_epsilon(
        batch_size=256,
        steps=1000,
        target_epsilon=target_epsilon,
        noise_range=(0.1, 5.0)
    )
    print(f"\\n目标: ε = {target_epsilon}")
    print(f"结果: 需要噪声乘子 σ = {noise:.3f}")
    print(f"验证: 使用 σ = {noise:.3f} 时，实际 ε = {actual_epsilon:.3f}")
    
    # 保存和绘图
    print("\n" + "=" * 60)
    print("保存配置和生成图表")
    print("=" * 60)
    
    accountant.save_config('privacy_config.json')
    accountant.plot_privacy_curve()
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
