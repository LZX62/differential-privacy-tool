import opacus.accountants.rdp as rdp_module

print("检查rdp_module类型:", type(rdp_module))
print("rdp_module的内容:", dir(rdp_module))

# 尝试从rdp模块的属性访问privacy_analysis
if hasattr(rdp_module, 'privacy_analysis'):
    print("\\nrdp_module有privacy_analysis属性")
    privacy_analysis = rdp_module.privacy_analysis
    print("privacy_analysis类型:", type(privacy_analysis))
    print("privacy_analysis的内容:", dir(privacy_analysis))
    
    # 尝试从privacy_analysis获取函数
    if hasattr(privacy_analysis, 'compute_rdp'):
        compute_rdp = privacy_analysis.compute_rdp
        print("✓ 成功获取compute_rdp函数")
    if hasattr(privacy_analysis, 'get_privacy_spent'):
        get_privacy_spent = privacy_analysis.get_privacy_spent
        print("✓ 成功获取get_privacy_spent函数")
else:
    print("\\nrdp_module没有privacy_analysis属性")

# 另一种方法：直接导入
print("\\n尝试直接导入privacy_analysis模块...")
try:
    import opacus.accountants.rdp.privacy_analysis as pa
    print("成功导入:", pa)
    print("模块路径:", pa.__file__)
except Exception as e:
    print("直接导入失败:", e)
