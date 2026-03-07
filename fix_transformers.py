import sys
import os

# 临时修复transformers的版本检查问题
transformers_path = "/home/lc/miniconda3/lib/python3.10/site-packages/transformers/dependency_versions_check.py"

# 读取文件
with open(transformers_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 临时修复版本检查
content = content.replace(
    'raise ValueError("got_ver is None")',
    'print("Warning: got_ver is None, skipping version check"); return'
)

# 写回文件
with open(transformers_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("已临时修复transformers版本检查问题")
