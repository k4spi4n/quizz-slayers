import json
import os
import re
import sys
import time
import tkinter as tk
from typing import Optional
from PIL import Image, ImageTk

from playwright.sync_api import Page
import litellm

# Add project root to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.core.auth import ensure_login_gui

LOGIN_URL = "https://edux.cmcu.edu.vn/login"
CONTEXT_PATH = os.path.join(os.path.dirname(__file__), "..", "context", "active-round.txt")

def sanitize_ai_response(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return text

def resolve_litellm_model(model: str, api_key: str) -> str:
    raw = model.strip()
    if "/" in raw:
        return raw
    if raw:
        if raw.startswith("gemini"):
            return f"gemini/{raw}"
        return raw
    key = api_key.strip()
    if key.startswith("AIza"):
        return "gemini/gemini-1.5-flash"
    if key.startswith("sk-ant"):
        return "anthropic/claude-3-5-sonnet-20240620"
    return "gpt-4o-mini"

def get_answer_via_litellm(question_text: str, choices: list, api_key: str, model: str) -> str:
    if not api_key:
        return ""
    
    choices_str = "\n".join([f"{c['key']}. {c['label']}" for c in choices])
    prompt = f"Question: {question_text}\nChoices:\n{choices_str}\n\nPick the correct answer key (e.g., A, B, true, false). Return ONLY the key, no explanation."
    
    model_name = resolve_litellm_model(model, api_key)
    try:
        response = litellm.completion(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            api_key=api_key,
            temperature=0,
        )
        content = response["choices"][0]["message"]["content"].strip()
        return sanitize_ai_response(content)
    except Exception as exc:
        print(f"[WARN] LiteLLM request failed: {exc}")
        return ""

def show_start_dialog(message: str) -> None:
    root = tk.Tk()
    root.title("EDUX Live Solver")
    root.attributes("-topmost", True)
    root.resizable(False, False)

    img_path = os.path.join(os.path.dirname(__file__), "..", "..", "EDUX-TEST-SOLVER", "img", "screen_to_start.png")
    if os.path.exists(img_path):
        try:
            pil_img = Image.open(img_path)
            w_percent = (280 / float(pil_img.size[0]))
            h_size = int((float(pil_img.size[1]) * float(w_percent)))
            pil_img = pil_img.resize((280, h_size), Image.Resampling.LANCZOS)
            img = ImageTk.PhotoImage(pil_img)
            img_label = tk.Label(root, image=img)
            img_label.image = img
            img_label.pack(pady=(10, 5), padx=10)
        except Exception as e:
            print(f"[WARN] Could not load image: {e}")

    label = tk.Label(root, text=message, wraplength=280, pady=5, font=("Segoe UI", 10))
    label.pack(padx=10)

    def on_start():
        root.destroy()

    start_button = tk.Button(root, text="Bắt đầu giám sát", command=on_start, width=20, height=1, font=("Segoe UI", 10, "bold"), bg="#4CAF50", fg="white")
    start_button.pack(pady=(5, 15))

    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    margin = 20
    x = screen_width - width - margin
    y = screen_height - height - margin - 40
    root.geometry(f"{width}x{height}+{x}+{y}")
    root.mainloop()

def test_live_solver(page: Page) -> None:
    email, password, api_key, model = ensure_login_gui()

    page.goto(LOGIN_URL, wait_until="domcontentloaded")
    
    if email and password:
        page.locator("#email").fill(email)
        page.locator("#password").fill(password)
        page.locator("#password").press("Enter")
        print("\n[INFO] Auto-login attempted.")
    else:
        print("\n[INFO] Vui lòng đăng nhập thủ công.")

    show_start_dialog("Mở trang lớp học trực tuyến, sau đó nhấn nút này để bắt đầu tự động trả lời.")

    print("[INFO] Monitoring active-round.txt...")
    last_question_id = None

    while not page.is_closed():
        if not os.path.exists(CONTEXT_PATH):
            time.sleep(0.5)
            continue

        try:
            with open(CONTEXT_PATH, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    time.sleep(0.2)
                    continue
                data_json = json.loads(content)
        except Exception as e:
            print(f"[ERROR] Failed to read/parse active-round.txt: {e}")
            time.sleep(0.5)
            continue

        if data_json.get("message") == "No active round" or not data_json.get("data"):
            time.sleep(0.5)
            continue

        data = data_json["data"]
        question_id = data.get("question_id")
        is_active = data.get("is_active")

        if is_active and question_id != last_question_id:
            print(f"\n[NEW QUESTION] ID: {question_id}")
            question_text = data.get("question_text", "")
            choices = data.get("choices", [])
            q_type = data.get("question_type")

            print(f"[INFO] Question: {question_text}")
            
            # 1. Get answer via AI
            answer_key = get_answer_via_litellm(question_text, choices, api_key, model)
            print(f"[AI ANSWER] {answer_key}")

            if not answer_key:
                print("[WARN] Could not get answer from AI.")
                last_question_id = question_id
                continue

            # 2. Pick answer in browser
            try:
                if q_type == "single_choice":
                    # Locate choice by text or key? The JSON has keys A, B, C, D.
                    # Usually the UI has buttons or divs.
                    # Based on test_solver.py:
                    # div.relative.flex.items-center.space-x-2.p-2.border.rounded-lg.cursor-pointer
                    # contains span.flex-shrink-0 for the letter (A, B, C, D)
                    
                    options = page.locator("div.relative.flex.items-center.space-x-2.p-2.border.rounded-lg.cursor-pointer")
                    count = options.count()
                    found = False
                    for i in range(count):
                        opt = options.nth(i)
                        letter_span = opt.locator("span.flex-shrink-0")
                        if letter_span.count() > 0:
                            letter = letter_span.inner_text().strip().replace(".", "")
                            if letter == answer_key:
                                opt.click()
                                print(f"[SUCCESS] Clicked option {answer_key}")
                                found = True
                                break
                    
                    if not found:
                        # Try searching by label text if key failed
                        target_label = ""
                        for c in choices:
                            if c["key"] == answer_key:
                                target_label = c["label"]
                                break
                        
                        if target_label:
                            for i in range(count):
                                opt = options.nth(i)
                                if target_label in opt.inner_text():
                                    opt.click()
                                    print(f"[SUCCESS] Clicked option with text matching AI answer")
                                    found = True
                                    break
                
                elif q_type == "true_false":
                    # true_false_blocks in test_solver were: div.border.border-gray-200.rounded-lg.p-3.bg-gray-50
                    # But in live view it might be different. Let's try buttons with "Đúng"/"Sai"
                    if answer_key.lower() in ["true", "đúng", "t"]:
                        page.get_by_role("button", name="Đúng").first.click()
                        print("[SUCCESS] Clicked 'Đúng'")
                    elif answer_key.lower() in ["false", "sai", "f"]:
                        page.get_by_role("button", name="Sai").first.click()
                        print("[SUCCESS] Clicked 'Sai'")
                
                last_question_id = question_id
            except Exception as e:
                print(f"[ERROR] Failed to pick answer: {e}")

        time.sleep(0.1) # Rapid polling
