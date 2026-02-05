# Opacus 1.5.4 的正确导入方式
import opacus.accountants.rdp as rdp_module
import numpy as np

# 通过属性访问获取函数
privacy_analysis = rdp_module.privacy_analysis
compute_rdp = privacy_analysis.compute_rdp
get_privacy_spent = privacy_analysis.get_privacy_spent

print("Opacus 1.5.4 隐私会计示例")
print("=" * 50)

# 1. 了解RDP orders（λ值）的概念
# orders是RDP的参数，通常取一系列值来找到最紧的边界
orders = np.arange(2, 100)  # λ从2到99

# 2. 计算RDP值
sampling_rate = 256 / 50000  # batch_size=256, dataset_size=50000
noise_multiplier = 1.0
steps = 1000

rdp = compute_rdp(
    q=sampling_rate,  # 采样率
    noise_multiplier=noise_multiplier,
    steps=steps,
    orders=orders
)

print(f"RDP值（前5个）: {rdp[:5]}")

# 3. 转换为(ε, δ)
delta = 1e-5

# 使用正确的参数名：delta（不是target_delta）
epsilon, optimal_alpha = get_privacy_spent(
    orders=orders, 
    rdp=rdp, 
    delta=delta
)

print(f"ε = {epsilon:.2f}")
print(f"最优α = {optimal_alpha:.1f}")
print(f"δ = {delta}")

# 额外分析
print(f"\\n额外分析:")
print(f"  要达到δ={delta}，隐私预算ε为{epsilon:.2f}")
print(f"  这意味着：在随机算法中，任何单个训练样本")
print(f"5. 在这个例子中，训练后达到({epsilon:.2f}, {delta})-差分隐私")