# verify_accounting.py
import numpy as np
from privacy_accountant_opacus import DPAccountant

def verify_parameter_effects():
    """验证参数对ε的影响规律"""
    accountant = DPAccountant(50000)
    
    print("=== 1. ε随训练步数增加而增加 ===")
    for steps in [100, 500, 1000, 2000]:
        eps, _, _ = accountant.compute_epsilon(256, steps, 1.0)
        print(f"Steps: {steps:4d} → ε: {eps:.2f}")
    
    print("\n=== 2. ε随噪声增加而减小 ===")
    for noise in [0.5, 1.0, 1.5, 2.0]:
        eps, _, _ = accountant.compute_epsilon(256, 1000, noise)
        print(f"Noise: {noise:.1f} → ε: {eps:.2f}")
    
    print("\n=== 3. ε随batch_size增加而增加 ===")
    for batch_size in [64, 128, 256, 512]:
        eps, _, _ = accountant.compute_epsilon(batch_size, 1000, 1.0)
        print(f"Batch: {batch_size:3d} → ε: {eps:.2f}")
    
    print("\n=== 4. 固定ε时，噪声与步数的权衡 ===")
    target_epsilon = 2.0
    steps_options = [500, 1000, 1500, 2000]
    
    for steps in steps_options:
        noise, actual_eps = accountant.compute_noise_for_target_epsilon(
            batch_size=256,
            steps=steps,
            target_epsilon=target_epsilon,
            noise_range=(0.1, 5.0)
        )
        print(f"Steps: {steps:4d} → 需要噪声σ: {noise:.3f} (实际ε: {actual_eps:.3f})")

def compare_with_manual_calculation():
    """与手动计算对比验证"""
    print("\n=== 与简单公式对比验证 ===")
    
    # 简单公式：ε ≈ √(steps) * sampling_rate / noise_multiplier
    accountant = DPAccountant(50000)
    
    batch_size = 256
    steps = 1000
    noise = 1.0
    
    # Opacus计算
    eps_opacus, delta, _ = accountant.compute_epsilon(batch_size, steps, noise)
    
    # 简单估算
    sampling_rate = batch_size / 50000
    eps_simple = np.sqrt(steps) * sampling_rate / noise
    
    print(f"Opacus计算: ε = {eps_opacus:.3f}")
    print(f"简单估算: ε ≈ {eps_simple:.3f}")
    print(f"相对误差: {abs(eps_opacus - eps_simple)/eps_opacus*100:.1f}%")

if __name__ == "__main__":
    verify_parameter_effects()
    compare_with_manual_calculation()