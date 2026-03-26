import os
import shutil
import tempfile
import time
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

# ========== 配置区 ==========
# 请修改为你想要抓取的ChatGPT对话URL
CHATGPT_URL = "https://chat.openai.com/chat"  # 或者你具体的对话URL
OUTPUT_DIR = "chat_md_exports"
SCROLL_PAUSE = 1.0
# ============================

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 创建临时目录来存放复制的Chrome数据
temp_data_dir = tempfile.mkdtemp(prefix="chrome_data_")
print(f"📁 创建临时Chrome数据目录: {temp_data_dir}")

# 复制你本地的Chrome数据到临时目录
source_dir = r"C:\Users\风之起兮漪于哞\AppData\Local\Google\Chrome\User Data\Default"
try:
    # 创建目标目录结构
    target_default_dir = os.path.join(temp_data_dir, "Default")
    os.makedirs(target_default_dir, exist_ok=True)
    
    # 只复制重要文件（避免复制大文件）
    important_files = [
        "Cookies", "Login Data", "Local State", 
        "Preferences", "Secure Preferences"
    ]
    
    for file_name in important_files:
        source_file = os.path.join(source_dir, file_name)
        target_file = os.path.join(target_default_dir, file_name)
        if os.path.exists(source_file):
            shutil.copy2(source_file, target_file)
            print(f"✓ 复制: {file_name}")
    
    print("✅ Chrome数据复制完成")
except Exception as e:
    print(f"⚠️ 数据复制失败，将使用空数据目录: {e}")

# 启动浏览器
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-gpu")

# 使用复制的数据目录
options.add_argument(f"--user-data-dir={temp_data_dir}")
options.add_argument("--profile-directory=Default")

# 添加这些参数避免常见错误
options.add_argument("--remote-debugging-port=9222")
options.add_argument("--disable-features=NetworkService")
options.add_argument("--log-level=3")
options.add_argument("--silent")

# 禁用自动化控制特征
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)

print("🚀 启动浏览器中...")

try:
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()), 
        options=options
    )
    
    print("✅ 浏览器启动成功！")
    
    # 打开ChatGPT
    print(f"正在打开: {CHATGPT_URL}")
    driver.get(CHATGPT_URL)
    time.sleep(5)  # 等待页面加载
    
    print(f"当前页面标题: {driver.title}")
    print(f"当前URL: {driver.current_url}")
    
    # 等待用户确认
    print("\n" + "="*60)
    print("请确认:")
    print("1. 页面已完全加载")
    print("2. 已显示你想要导出的对话")
    print("3. 对话内容已滚动到顶部")
    print("="*60)
    
    input("确认无误后按回车键开始抓取对话内容...")
    
    # 首先滚动到顶部，确保从头开始抓取
    print("正在滚动到页面顶部...")
    driver.execute_script("window.scrollTo(0, 0);")
    time.sleep(2)
    
    # 逐步向下滚动，加载所有内容
    print("正在加载所有对话内容...")
    last_height = driver.execute_script("return document.body.scrollHeight")
    scroll_count = 0
    
    while scroll_count < 50:  # 最多滚动50次
        # 滚动到底部
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE)
        
        # 检查是否还有更多内容
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            # 尝试点击可能的"加载更多"按钮
            try:
                load_more_buttons = driver.find_elements(By.XPATH, "//button[contains(text(), 'Load') or contains(text(), '加载')]")
                for button in load_more_buttons:
                    try:
                        button.click()
                        time.sleep(2)
                        print("✓ 点击了加载更多按钮")
                    except:
                        pass
            except:
                pass
            
            # 再次检查高度
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                print(f"✅ 所有内容已加载，共滚动 {scroll_count} 次")
                break
        
        last_height = new_height
        scroll_count += 1
        
        if scroll_count % 5 == 0:
            print(f"  已滚动 {scroll_count} 次，当前页面高度: {new_height}")
    
    # 回到顶部开始抓取
    driver.execute_script("window.scrollTo(0, 0);")
    time.sleep(2)
    
    # 尝试不同的选择器来抓取对话
    print("\n正在抓取对话内容...")
    
    # 尝试多种可能的选择器
    selectors_to_try = [
        # ChatGPT 4o 可能的选择器
        "div[data-message-author-role]",
        "div.group",  # 原始选择器
        "div[class*='markdown']",
        "div.text-gray-800",  # 用户消息
        "div.text-gray-600",  # AI消息
        "article",  # 可能用article标签
        "div.prose",  # 文本内容
    ]
    
    all_messages = []
    
    for selector in selectors_to_try:
        try:
            elements = driver.find_elements(By.CSS_SELECTOR, selector)
            if elements:
                print(f"使用选择器 '{selector}' 找到 {len(elements)} 个元素")
                for elem in elements:
                    try:
                        text = elem.text.strip()
                        if text and len(text) > 10:  # 过滤掉太短的文本
                            # 尝试判断是用户还是AI
                            role = "assistant"
                            parent_html = elem.get_attribute("outerHTML")
                            if "user" in parent_html.lower() or "你" in text[:20] or "You" in text[:20]:
                                role = "user"
                            
                            all_messages.append({
                                "role": role,
                                "content": text,
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
                    except:
                        continue
                
                if len(all_messages) > 0:
                    print(f"✓ 成功抓取到 {len(all_messages)} 条消息")
                    break
        except:
            continue
    
    # 如果上述选择器都不行，尝试更通用的方法
    if not all_messages:
        print("尝试通用文本抓取...")
        all_text_elements = driver.find_elements(By.XPATH, "//div[text()]")
        conversations = []
        current_conversation = ""
        
        for elem in all_text_elements:
            text = elem.text.strip()
            if len(text) > 20:  # 只保留较长的文本
                conversations.append(text)
        
        # 简单分组：假设交替出现用户和AI消息
        for i, text in enumerate(conversations):
            role = "user" if i % 2 == 0 else "assistant"
            all_messages.append({
                "role": role,
                "content": text,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
    
    # 去重并排序
    unique_messages = []
    seen_content = set()
    
    for msg in all_messages:
        if msg["content"] not in seen_content and len(msg["content"]) > 10:
            seen_content.add(msg["content"])
            unique_messages.append(msg)
    
    # 按页面中的位置排序（简单的基于角色交替的排序）
    user_messages = [m for m in unique_messages if m["role"] == "user"]
    assistant_messages = [m for m in unique_messages if m["role"] == "assistant"]
    
    # 简单交替合并
    final_conversation = []
    max_len = max(len(user_messages), len(assistant_messages))
    
    for i in range(max_len):
        if i < len(user_messages):
            final_conversation.append(user_messages[i])
        if i < len(assistant_messages):
            final_conversation.append(assistant_messages[i])
    
    # 保存到Markdown文件
    if final_conversation:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        md_filename = os.path.join(OUTPUT_DIR, f"chat_{timestamp_str}.md")
        
        with open(md_filename, "w", encoding="utf-8") as f:
            f.write(f"# ChatGPT 对话导出\n\n")
            f.write(f"**导出时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**页面标题**: {driver.title}\n\n")
            f.write(f"**页面URL**: {driver.current_url}\n\n")
            f.write("---\n\n")
            
            for i, msg in enumerate(final_conversation, 1):
                role_zh = "用户" if msg["role"] == "user" else "ChatGPT"
                f.write(f"## {role_zh} (消息 {i})\n\n")
                f.write(f"{msg['content']}\n\n")
                f.write("---\n\n")
        
        print(f"\n🎉 导出成功！")
        print(f"📄 文件保存至: {os.path.abspath(md_filename)}")
        print(f"📊 共导出 {len(final_conversation)} 条消息")
        
        # 在Windows中打开文件所在目录
        try:
            os.startfile(os.path.dirname(os.path.abspath(md_filename)))
        except:
            print(f"📁 文件目录: {os.path.dirname(os.path.abspath(md_filename))}")
    else:
        print("❌ 未找到任何对话内容")
        
        # 保存页面HTML用于调试
        debug_filename = os.path.join(OUTPUT_DIR, f"debug_page_{timestamp_str}.html")
        with open(debug_filename, "w", encoding="utf-8") as f:
            f.write(driver.page_source)
        print(f"💡 已保存页面HTML到: {debug_filename}")
        print("   你可以用浏览器打开这个文件查看页面结构")
    
    # 关闭浏览器
    driver.quit()
    print("✅ 浏览器已关闭")
    
except Exception as e:
    print(f"❌ 发生错误: {str(e)}")
    import traceback
    traceback.print_exc()
    
finally:
    # 清理临时目录
    try:
        time.sleep(1)
        shutil.rmtree(temp_data_dir)
        print(f"🧹 已清理临时目录: {temp_data_dir}")
    except Exception as e:
        print(f"⚠️ 清理临时目录失败: {e}")

print("\n✨ 脚本执行完成！")