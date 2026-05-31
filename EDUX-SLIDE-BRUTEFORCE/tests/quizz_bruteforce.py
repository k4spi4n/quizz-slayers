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


def extract_answer_texts(answers_locator) -> list[str]:
        return answers_locator.evaluate_all(
                """
                nodes => nodes.map(node => {
                    const letter = node.querySelector('span.font-bold')?.innerText?.trim() || '';
                    const text = node.querySelector('div.prose p')?.innerText?.trim() || '';
                    return `${letter} ${text}`.trim();
                })
                """
        )


def test_wait_for_user_login(page: Page) -> None:
    email, password = ensure_login_gui()
    page.goto(LOGIN_URL, wait_until="domcontentloaded")

    if email and password:
        page.locator("#email").fill(email)
        page.locator("#password").fill(password)
        page.locator("#password").press("Enter")
        print("\n[INFO] Auto-login attempted. If needed, finish any extra steps in the browser.")
    else:
        print("\n[INFO] 'Tự đăng nhập' được chọn. Vui lòng đăng nhập thủ công trên trình duyệt.")
    
    show_start_dialog("Khi bạn thấy màn hình slide, chuyển tới slide đang làm mới nhất và nhấn nút dưới đây để bắt đầu tự động trả lời.")

    wrong_answers: dict[str, set[str]] = {}
    question_answer_cache: dict[str, list[str]] = {}

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
            print("[INFO] No question on this slide. Clicked 'Trang sau'.")
            page.wait_for_load_state("domcontentloaded")
            page.wait_for_timeout(200)
            continue

        if not question_locator.is_visible():
            answer_button.wait_for(state="visible")
            answer_button.click()

        try:
            question_locator.wait_for(state="visible", timeout=10000)
        except Exception:
            print("[WARN] Question not visible yet. Retrying loop.")
            page.wait_for_timeout(200)
            continue

        question_text = question_locator.inner_text().strip()
        print(f"[INFO] Question: {question_text}")

        answer_texts = question_answer_cache.get(question_text)
        if answer_texts is None:
            try:
                answers_locator.first.wait_for(state="visible", timeout=10000)
            except Exception:
                print("[WARN] Answers not visible yet. Retrying loop.")
                page.wait_for_timeout(200)
                continue

            answer_texts = extract_answer_texts(answers_locator)
            question_answer_cache[question_text] = answer_texts

        answer_count = len(answer_texts)
        print(f"[INFO] Answers found: {answer_count}")

        target_answer = os.environ.get("AUTO_ANSWER_TEXT", "").strip()
        clicked_answer = False
        chosen_answer_text = ""
        if target_answer:
            print(f"[INFO] Auto-answer target: {target_answer}")
            match = answers_locator.filter(has_text=target_answer).first
            match.click()
            clicked_answer = True
            chosen_answer_text = target_answer
        elif answer_count > 0:
            tried_for_question = wrong_answers.get(question_text, set())
            next_index = next(
                (i for i, text in enumerate(answer_texts) if text not in tried_for_question),
                None,
            )
            if next_index is None:
                tried_for_question.clear()
                next_index = 0

            chosen_answer_text = answer_texts[next_index]
            print(f"[INFO] AUTO_ANSWER_TEXT not set. Pick: {chosen_answer_text}")
            answers_locator.nth(next_index).click()
            clicked_answer = True
        else:
            print("[WARN] No answers available to click.")

        if clicked_answer:
            check_button.click()
            print("[INFO] Clicked 'Kiem tra' button.")

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
                    print("[INFO] Clicked 'Trang sau' button.")
                elif next_button.is_visible():
                    next_button.click()
                    print("[INFO] Clicked 'Cau tiep theo' button.")
                elif retry_button.is_visible():
                    retry_button.click()
                    if chosen_answer_text:
                        wrong_answers.setdefault(question_text, set()).add(chosen_answer_text)
                        print(f"[INFO] Marked wrong answer: {chosen_answer_text}")
                    print("[INFO] Clicked 'Thu lai' button.")
                else:
                    print("[WARN] Follow-up buttons not visible after wait.")
            except Exception:
                print("[WARN] No follow-up button appeared.")

        page.wait_for_timeout(200)

    # Keep this test non-failing while we are still wiring selectors.
    print(f"[INFO] Current URL after click: {page.url}")
    print("[INFO] Browser will stay open. Close the browser window to finish.")
    page.wait_for_event("close")
