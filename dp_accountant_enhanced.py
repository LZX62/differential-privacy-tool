# dp_accountant_enhanced.py
import sys
import os

# 添加当前目录到路径，确保可以导入第4天的代码
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from privacy_accountant_opacus import DPAccountant  # 导入第4天的类
except ImportError:
    print("错误: 找不到 privacy_accountant_opacus.py，请确保第4天的代码在同一目录")
    print("你可以从以下代码创建该文件：")
    print("# privacy_accountant_opacus.py 的内容...")
    sys.exit(1)

import numpy as np
from datetime import datetime
import json

class EnhancedDPAccountant(DPAccountant):
    """增强版隐私会计工具，继承自第4天的DPAccountant"""
    
    def __init__(self, dataset_size, delta_method='1/N'):
        super().__init__(dataset_size, delta_method)
        self.enhanced_history = []  # 额外记录增强功能的历史
    
    def generate_epsilon_grid(self, target_epsilons, batch_size, steps, 
                             clip_norm=1.0, epsilon_tolerance=0.1):
        """
        为目标ε网格生成对应的噪声乘子配置
        
        Args:
            target_epsilons: 目标ε列表，如[0.5, 1, 2, 4, 8]
            batch_size: 批次大小
            steps: 训练步数
            clip_norm: 裁剪阈值
            epsilon_tolerance: ε容忍度
            
        Returns:
            dict: {epsilon: noise_multiplier} 映射
        """
        configs = {}
        
        for target_eps in target_epsilons:
            noise, actual_eps = self.compute_noise_for_target_epsilon(
                batch_size=batch_size,
                steps=steps,
                target_epsilon=target_eps,
                tolerance=epsilon_tolerance
            )
            configs[target_eps] = {
                'noise_multiplier': noise,
                'actual_epsilon': actual_eps,
                'delta': self.compute_delta(),
                'batch_size': batch_size,
                'steps': steps,
                'clip_norm': clip_norm
            }
            print(f"ε={target_eps:.1f} → σ={noise:.3f} (实际ε={actual_eps:.3f})")
        
        return configs
    
    def compare_delta_methods(self, batch_size, steps, noise_multiplier):
        """比较不同δ计算方法对ε的影响"""
        # 保存原始方法
        original_method = self.delta_method
        
        methods = ['1/N', '1/N^1.1', 'fixed:1e-5']
        results = []
        
        for method in methods:
            self.delta_method = method  # 临时改变计算方法
            eps, delta, alpha = self.compute_epsilon(
                batch_size=batch_size,
                steps=steps,
                noise_multiplier=noise_multiplier
            )
            results.append({
                'method': method,
                'epsilon': eps,
                'delta': delta,
                'optimal_alpha': alpha
            })
        
        # 恢复原始方法
        self.delta_method = original_method
        
        return results
    
    def batch_analyze(self, param_grid):
        """
        批量分析不同参数组合
        
        Args:
            param_grid: 参数网格，例如:
                {
                    'batch_sizes': [64, 128, 256],
                    'noise_multipliers': [0.5, 1.0, 2.0],
                    'steps_list': [100, 500, 1000]
                }
        """
        results = []
        
        # 检查参数网格格式
        required_keys = ['batch_sizes', 'noise_multipliers', 'steps_list']
        for key in required_keys:
            if key not in param_grid:
                raise ValueError(f"参数网格缺少必需的键: {key}")
        
        total_combinations = (len(param_grid['batch_sizes']) * 
                             len(param_grid['noise_multipliers']) * 
                             len(param_grid['steps_list']))
        print(f"开始批量分析: 共 {total_combinations} 种参数组合")
        
        count = 0
        for batch_size in param_grid['batch_sizes']:
            for noise_multiplier in param_grid['noise_multipliers']:
                for steps in param_grid['steps_list']:
                    eps, delta, alpha = self.compute_epsilon(
                        batch_size=batch_size,
                        steps=steps,
                        noise_multiplier=noise_multiplier
                    )
                    results.append({
                        'batch_size': batch_size,
                        'noise_multiplier': noise_multiplier,
                        'steps': steps,
                        'epsilon': eps,
                        'delta': delta,
                        'optimal_alpha': alpha
                    })
                    count += 1
                    if count % 10 == 0:
                        print(f"进度: {count}/{total_combinations}")
        
        print(f"批量分析完成！共分析 {count} 种参数组合")
        return results
    
    def export_results_to_csv(self, results, filename='dp_analysis_results.csv'):
        """将分析结果导出为CSV文件"""
        import csv
        
        if not results:
            print("没有结果可导出")
            return False
        
        # 定义CSV列名
        fieldnames = ['batch_size', 'noise_multiplier', 'steps', 
                     'epsilon', 'delta', 'optimal_alpha']
        
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            print(f"结果已导出到 {filename}")
            return True
        except Exception as e:
            print(f"导出CSV时出错: {e}")
            return False
    
    def analyze_parameter_sensitivity(self, base_params, variations=0.1):
        """
        分析参数敏感性
        
        Args:
            base_params: 基础参数，包含batch_size, steps, noise_multiplier
            variations: 变化幅度，如0.1表示±10%
        """
        base_eps, base_delta, base_alpha = self.compute_epsilon(**base_params)
        
        sensitivity_results = {
            'base': {
                'parameters': base_params,
                'epsilon': base_eps,
                'delta': base_delta,
                'alpha': base_alpha
            },
            'variations': []
        }
        
        # 分析每个参数的敏感性
        for param_name in ['batch_size', 'steps', 'noise_multiplier']:
            # 增加10%
            params_plus = base_params.copy()
            params_plus[param_name] = params_plus[param_name] * (1 + variations)
            eps_plus, _, _ = self.compute_epsilon(**params_plus)
            
            # 减少10%
            params_minus = base_params.copy()
            params_minus[param_name] = params_minus[param_name] * (1 - variations)
            eps_minus, _, _ = self.compute_epsilon(**params_minus)
            
            sensitivity_change = (eps_plus - eps_minus) / (2 * base_eps)
            
            sensitivity_results['variations'].append({
                'parameter': param_name,
                'increase_10%': eps_plus,
                'decrease_10%': eps_minus,
                'sensitivity': sensitivity_change
            })
            
            print(f"{param_name}: ε变化 {sensitivity_change:.1%} (±{variations*100}%)")
        
        return sensitivity_results


# 测试增强功能
if __name__ == "__main__":
    print("=== 隐私会计工具增强版测试 ===\n")
    
    # 创建增强版会计器
    try:
        accountant = EnhancedDPAccountant(dataset_size=50000)
        print(f"成功创建会计器，数据集大小: {accountant.dataset_size}")
    except Exception as e:
        print(f"创建会计器失败: {e}")
        sys.exit(1)
    
    print("\n=== 测试ε网格生成功能 ===")
    try:
        epsilons = [0.5, 1.0, 2.0, 4.0, 8.0]
        configs = accountant.generate_epsilon_grid(
            target_epsilons=epsilons,
            batch_size=256,
            steps=10000,
            clip_norm=1.0
        )
        print(f"成功生成 {len(configs)} 个ε配置")
    except Exception as e:
        print(f"ε网格生成失败: {e}")
    
    print("\n=== 测试δ方法对比 ===")
    try:
        comparisons = accountant.compare_delta_methods(
            batch_size=256,
            steps=1000,
            noise_multiplier=1.0
        )
        
        for comp in comparisons:
            print(f"δ方法: {comp['method']:10} → ε={comp['epsilon']:.2f}, δ={comp['delta']:.2e}")
    except Exception as e:
        print(f"δ方法对比失败: {e}")
    
    print("\n=== 测试批量分析功能 ===")
    try:
        param_grid = {
            'batch_sizes': [64, 128, 256],
            'noise_multipliers': [0.5, 1.0, 2.0],
            'steps_list': [100, 500, 1000]
        }
        
        batch_results = accountant.batch_analyze(param_grid)
        print(f"批量分析完成，共 {len(batch_results)} 条结果")
        
        # 导出为CSV
        accountant.export_results_to_csv(batch_results, 'batch_analysis.csv')
    except Exception as e:
        print(f"批量分析失败: {e}")
    
    print("\n=== 测试参数敏感性分析 ===")
    try:
        base_params = {
            'batch_size': 256,
            'steps': 1000,
            'noise_multiplier': 1.0
        }
        
        sensitivity = accountant.analyze_parameter_sensitivity(base_params, variations=0.1)
        print("参数敏感性分析完成")
    except Exception as e:
        print(f"敏感性分析失败: {e}")
    
    # 保存配置
    try:
        with open('enhanced_configs.json', 'w') as f:
            json.dump({
                'epsilon_grid': configs if 'configs' in locals() else {},
                'delta_comparisons': comparisons if 'comparisons' in locals() else [],
                'batch_results_count': len(batch_results) if 'batch_results' in locals() else 0,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2, default=str)
        
        print("\n配置已保存到 enhanced_configs.json")
    except Exception as e:
        print(f"保存配置失败: {e}")
    
    print("\n=== 所有测试完成 ===")