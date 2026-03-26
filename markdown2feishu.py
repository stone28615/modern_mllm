import re
import sys

def process_latex_in_markdown(md_text: str) -> str:
    """
    对 Markdown 文本中的 LaTeX 公式执行以下处理：
    1. "$$"  --> "syc19253"
    2. "$"   --> " $ "
    3. "syc19253" --> "$$"
    4. 去掉所有的 "\\n" 换行
    同时保护代码块（```...```）和行内代码（`...`）中的 $ 不被处理。
    """
    placeholder = "syc19253"
    code_blocks = []
    inline_codes = []

    # Step A: 提取并临时替换代码块（```...```），支持跨行
    def replace_code_block(match):
        code_blocks.append(match.group(0))
        return f"__CODE_BLOCK_{len(code_blocks) - 1}__"
    
    # 使用非贪婪匹配，支持多语言标识如 ```python
    md_text = re.sub(r'```[\s\S]*?```', replace_code_block, md_text)

    # Step B: 提取并临时替换行内代码（`...`）
    def replace_inline_code(match):
        inline_codes.append(match.group(0))
        return f"__INLINE_CODE_{len(inline_codes) - 1}__"
    
    md_text = re.sub(r'`[^`]*`', replace_inline_code, md_text)

    # Step C: 在非代码区域执行四步公式处理
    # 1. "$$" → 临时占位符
    md_text = md_text.replace("$$", placeholder)
    # 2. "$" → " $ "（添加空格）
    md_text = md_text.replace("$", " $ ")
    # 3. 占位符 → "$$"
    md_text = md_text.replace(placeholder, "$$")
    # 4. 移除所有换行符
    md_text = md_text.replace("\n", "")

    # Step D: 恢复行内代码
    def restore_inline_code(match):
        idx = int(match.group(1))
        return inline_codes[idx]
    
    md_text = re.sub(r'__INLINE_CODE_(\d+)__', restore_inline_code, md_text)

    # Step E: 恢复代码块
    def restore_code_block(match):
        idx = int(match.group(1))
        return code_blocks[idx]
    
    md_text = re.sub(r'__CODE_BLOCK_(\d+)__', restore_code_block, md_text)

    return md_text


# ======================
# 如果作为脚本直接运行，从 stdin 读取或从文件读取
# ======================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process LaTeX formulas in Markdown text.")
    parser.add_argument("-i", "--input", help="Input Markdown file path. If not provided, read from stdin.")
    parser.add_argument("-o", "--output", help="Output file path. If not provided, write to stdout.")

    args = parser.parse_args()

    # 读取输入
    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            input_text = f.read()
    else:
        input_text = sys.stdin.read()

    # 处理
    output_text = process_latex_in_markdown(input_text)

    # 输出
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_text)
    else:
        sys.stdout.write(output_text)