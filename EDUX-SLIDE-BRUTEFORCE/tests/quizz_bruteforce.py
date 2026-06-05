import os
import random
import tkinter as tk
import sys
from PIL import Image, ImageTk

from playwright.sync_api import Page

# Add project root to sys.path to import shared modules
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.core.auth import ensure_login_gui

LOGIN_URL = "https://edux.cmcu.edu.vn/login"


def show_start_dialog(message: str) -> None:
    root = tk.Tk()
    root.title("Sẵn sàng?")
    root.attributes("-topmost", True)
    root.resizable(False, False)

    # Load and resize image
    img_path = os.path.join(os.path.dirname(__file__), "..", "img", "screen_to_start.png")
    if os.path.exists(img_path):
        try:
            pil_img = Image.open(img_path)
            # Resize to width 280, maintain aspect ratio
            w_percent = (280 / float(pil_img.size[0]))
            h_size = int((float(pil_img.size[1]) * float(w_percent)))
            pil_img = pil_img.resize((280, h_size), Image.Resampling.LANCZOS)
            
            img = ImageTk.PhotoImage(pil_img)
            img_label = tk.Label(root, image=img)
            img_label.image = img  # Keep reference
            img_label.pack(pady=(10, 5), padx=10)
        except Exception as e:
            print(f"[WARN] Could not load image: {e}")

    label = tk.Label(root, text=message, wraplength=280, pady=5, font=("Segoe UI", 10))
    label.pack(padx=10)

    def on_start():
        root.destroy()

    start_button = tk.Button(root, text="Bắt đầu ngay", command=on_start, width=20, height=1, font=("Segoe UI", 10, "bold"), bg="#4CAF50", fg="white")
    start_button.pack(pady=(5, 15))

    # Position at bottom right
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    
    # Calculate x, y for bottom right with a small margin
    margin = 20
    x = screen_width - width - margin
    y = screen_height - height - margin - 40 # -40 for taskbar
    
    root.geometry(f"{width}x{height}+{x}+{y}")
    root.mainloop()


def test_wait_for_user_login(page: Page) -> None:
    email, password, _api_key, _model = ensure_login_gui()
    page.goto(LOGIN_URL, wait_until="domcontentloaded")

    if email and password:
        page.locator("#email").fill(email)
        page.locator("#password").fill(password)
        page.locator("#password").press("Enter")
        print("\n[INFO] Auto-login attempted. If needed, finish any extra steps in the browser.")
    else:
        print("\n[INFO] 'Tự đăng nhập' được chọn. Vui lòng đăng nhập thủ công trên trình duyệt.")
    
    show_start_dialog("Khi bạn thấy màn hình slide, chuyển tới slide đang làm mới nhất và nhấn nút dưới đây để bắt đầu tự động trả lời.")

    wrong_answers: dict[str, set[int]] = {}

    no_question_button = page.get_by_role("button", name="Không có câu hỏi")
    answer_button = page.get_by_role("button", name="Trả lời trên lớp")
    check_button = page.get_by_role("button", name="Kiểm tra")
    next_button = page.get_by_role("button", name="Câu tiếp theo")
    retry_button = page.get_by_role("button", name="Thử lại")
    next_page_button = page.get_by_role("button", name="Trang sau")
    question_locator = page.locator("p.my-3.text-gray-800.leading-relaxed").first
    answers_locator = page.locator(
        "div.flex.items-center.space-x-6.p-8.rounded-xl.border-2.transition-colors.cursor-pointer.min-h-\\[80px\\]"
    )

    while not page.is_closed():
        if no_question_button.is_visible():
            next_page_button.click()
            print("[INFO] Next Page (No Question)")
            try: no_question_button.wait_for(state="hidden", timeout=1000)
            except: pass
            continue

        if not question_locator.is_visible():
            if answer_button.is_visible():
                answer_button.click()
            else:
                page.wait_for_timeout(500)
                continue

        try:
            question_locator.wait_for(state="visible", timeout=5000)
        except Exception:
            continue

        question_text = question_locator.inner_text().strip()
        print(f"\n[Q] {question_text[:60]}...")

        try:
            answers_locator.first.wait_for(state="visible", timeout=5000)
        except:
            continue

        answer_count = answers_locator.count()
        if answer_count == 0:
            continue

        tried_indices = wrong_answers.get(question_text, set())
        # Reset if we tried all
        if len(tried_indices) >= answer_count:
            tried_indices.clear()

        next_index = next((i for i in range(answer_count) if i not in tried_indices), 0)
        
        print(f"[Pick] #{next_index + 1}/{answer_count}")
        answers_locator.nth(next_index).click()
        check_button.click()

        try:
            page.wait_for_function(
                """
                () => {
                  const labels = ['Trang sau', 'Câu tiếp theo', 'Thử lại'];
                  return labels.some(label => {
                    const btn = Array.from(document.querySelectorAll('button'))
                      .find(b => (b.textContent || '').trim() === label);
                    return btn && !btn.disabled && btn.offsetParent !== null;
                  });
                }
                """,
                timeout=10000,
            )

            if next_page_button.is_visible():
                next_page_button.click()
                print("[Done] Next Page")
                try: next_page_button.wait_for(state="hidden", timeout=1000)
                except: pass
            elif next_button.is_visible():
                next_button.click()
                print("[Done] Next Question")
                try: next_button.wait_for(state="hidden", timeout=1000)
                except: pass
            elif retry_button.is_visible():
                wrong_answers.setdefault(question_text, set()).add(next_index)
                print(f"[Wrong] Index {next_index + 1} marked")
                retry_button.click()
                try: retry_button.wait_for(state="hidden", timeout=1000)
                except: pass
        except Exception:
            print("[WARN] No follow-up button")

    page.wait_for_event("close")
