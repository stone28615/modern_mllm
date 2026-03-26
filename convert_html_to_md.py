import os
import re
import email
from bs4 import BeautifulSoup
from datetime import datetime

# ========= 配置 =========
MHTML_FILE = "chatgpt_conversation.mhtml"
OUTPUT_DIR = "chat_md_exports"
CALLOUT_TYPE = "note"
# =======================

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_html_from_mhtml(path):
    with open(path, "rb") as f:
        msg = email.message_from_bytes(f.read())
    for part in msg.walk():
        if part.get_content_type() == "text/html":
            return part.get_payload(decode=True).decode("utf-8", "ignore")
    raise RuntimeError("未找到 HTML")

def preserve_latex(text):
    text = re.sub(r'\$\$(.+?)\$\$', r'\n$$\n\1\n$$\n', text, flags=re.S)
    text = re.sub(r'(?<!\$)\$(.+?)\$(?!\$)', r'$\1$', text)
    return text

def extract_blocks(html):
    soup = BeautifulSoup(html, "lxml")
    blocks = []

    message_divs = soup.find_all("div", attrs={"data-message-author-role": True})

    for div in message_divs:
        role = div["data-message-author-role"]  # user / assistant

        # 移除无关元素
        for tag in div.find_all(["button", "svg", "path"]):
            tag.decompose()

        text = div.get_text("\n", strip=True)
        text = preserve_latex(text)
        text = re.sub(r'\n{3,}', '\n\n', text)

        if text:
            blocks.append((role, text))

    return blocks

def build_qa(blocks):
    qa = []
    current = None

    for role, text in blocks:
        if role == "user":
            current = {"q": text, "a": []}
            qa.append(current)
        elif role == "assistant" and current:
            current["a"].append(text)

    return qa

def write_md(qa):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUTPUT_DIR, f"chat_{ts}.md")

    with open(path, "w", encoding="utf-8") as f:
        f.write("# ChatGPT 学习对话（结构化导出）\n\n")
        f.write(f"- 导出时间：{datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
        f.write("---\n\n")

        for i, item in enumerate(qa, 1):
            f.write(f"## 问题 {i}\n\n")
            for line in item["q"].splitlines():
                f.write(f"> {line}\n")
            f.write("\n")

            for ans in item["a"]:
                f.write(f"> [!{CALLOUT_TYPE}] ChatGPT · 回答\n")
                for line in ans.splitlines():
                    f.write(f"> {line}\n")
                f.write("\n")

            f.write("---\n\n")

    return path

def main():
    html = extract_html_from_mhtml(MHTML_FILE)
    blocks = extract_blocks(html)

    if not blocks:
        print("❌ 未提取到任何消息")
        return

    qa = build_qa(blocks)
    if not qa:
        print("❌ 未形成问答结构")
        return

    out = write_md(qa)
    print(f"✅ 成功导出：{out}")

if __name__ == "__main__":
    main()
