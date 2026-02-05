import opacus.accountants.rdp as rdp_module
privacy_analysis = rdp_module.privacy_analysis
get_privacy_spent = privacy_analysis.get_privacy_spent

# 使用inspect模块查看函数签名
import inspect
signature = inspect.signature(get_privacy_spent)
print("get_privacy_spent函数签名:")
print(signature)

print("\\n参数详情:")
for param_name, param in signature.parameters.items():
    print(f"  {param_name}: {param.default if param.default != inspect.Parameter.empty else '无默认值'}")
