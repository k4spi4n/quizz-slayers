import os
import random
import time
import tkinter as tk
import sys
from PIL import Image, ImageTk

from playwright.sync_api import (
    Page,
    Error as PlaywrightError,
    TimeoutError as PlaywrightTimeoutError,
)

# Add project root to sys.path to import shared modules
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.core.auth import ensure_login_gui

LOGIN_URL = "https://edux.cmcu.edu.vn/login"

# --- Tham số chịu lỗi / chịu lag mạng (giữ tốc độ cao) ---
# Timeout cho mỗi thao tác click: đủ dài để vượt qua jitter mạng, đủ ngắn để
# không treo cả vòng lặp khi element thật sự không tồn tại.
ACTION_TIMEOUT_MS = 7000
# Thời gian nghỉ ngắn khi gặp lỗi tạm thời trước khi thử lại vòng lặp.
TRANSIENT_BACKOFF_MS = 250


def safe_is_visible(locator) -> bool:
    """is_visible() nhưng nuốt lỗi tạm thời (context bị huỷ khi điều hướng, v.v.)."""
    try:
        return locator.is_visible()
    except PlaywrightError:
        return False


def safe_click(locator, timeout: int = ACTION_TIMEOUT_MS) -> bool:
    """Click có auto-wait + nuốt lỗi tạm thời. Trả về True nếu click thành công."""
    try:
        locator.click(timeout=timeout)
        return True
    except PlaywrightError:
        return False


def safe_goto(page: Page, url: str, attempts: int = 5) -> bool:
    """goto có retry để chịu được lag/lỗi mạng lúc tải trang đầu."""
    for i in range(attempts):
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            return True
        except PlaywrightError as e:
            print(f"[WARN] goto thất bại (lần {i + 1}/{attempts}): {str(e)[:80]}")
            page.wait_for_timeout(1000)
    return False


def answers_fingerprint(answers_locator) -> str:
    """Khoá ghi nhớ dựa trên nội dung các đáp án — dùng khi không lấy được text câu hỏi."""
    try:
        texts = answers_locator.all_inner_texts()
        joined = " | ".join(t.strip() for t in texts if t.strip())
        return joined[:200]
    except PlaywrightError:
        return ""


def log_stall_diagnostics(page: Page, buttons: dict) -> None:
    """In ra trạng thái màn hình khi nghi bị kẹt, để biết LÝ DO thay vì im lặng."""
    visible = [name for name, loc in buttons.items() if safe_is_visible(loc)]
    try:
        url = page.url
    except PlaywrightError:
        url = "?"
    print(
        f"[STALL] Đang chờ (chưa thấy đáp án). "
        f"Nút đang hiện: {visible or 'không có'} | URL: {url}"
    )


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

    # Timeout mặc định cho mọi thao tác: auto-wait sẽ tự thử lại trong khoảng này,
    # giúp vượt qua lag mạng mà không treo vô hạn.
    page.set_default_timeout(ACTION_TIMEOUT_MS)

    if not safe_goto(page, LOGIN_URL):
        print("[ERROR] Không tải được trang đăng nhập sau nhiều lần thử. Dừng lại.")
        return

    if email and password:
        try:
            page.locator("#email").fill(email)
            page.locator("#password").fill(password)
            page.locator("#password").press("Enter")
            print("\n[INFO] Auto-login attempted. If needed, finish any extra steps in the browser.")
        except PlaywrightError as e:
            print(f"\n[WARN] Tự đăng nhập gặp lỗi ({str(e)[:80]}). Vui lòng đăng nhập thủ công.")
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

    # Watchdog phát hiện kẹt: nếu quá lâu không thấy đáp án nào để trả lời thì
    # in chẩn đoán (1 lần mỗi lần kẹt) để biết lý do thay vì dừng im lặng.
    all_buttons = {
        "Không có câu hỏi": no_question_button,
        "Trả lời trên lớp": answer_button,
        "Kiểm tra": check_button,
        "Câu tiếp theo": next_button,
        "Thử lại": retry_button,
        "Trang sau": next_page_button,
    }
    last_progress = time.monotonic()
    stall_reported = False
    STALL_SECONDS = 8.0

    while not page.is_closed():
        # Mỗi vòng lặp được cô lập: lỗi tạm thời (mất mạng, context bị huỷ khi
        # điều hướng, element detach do re-render) chỉ làm bỏ qua 1 vòng rồi thử
        # lại, KHÔNG làm sập cả script.
        try:
            if safe_is_visible(no_question_button):
                safe_click(next_page_button)
                print("[INFO] Next Page (No Question)")
                try: no_question_button.wait_for(state="hidden", timeout=1000)
                except PlaywrightError: pass
                last_progress = time.monotonic(); stall_reported = False
                continue

            # Tín hiệu CHÍNH để trả lời: các đáp án đã hiện hay chưa.
            # (Không phụ thuộc vào selector text câu hỏi vốn dễ vỡ.)
            answers_visible = safe_is_visible(answers_locator.first)

            if not answers_visible:
                # Chưa có đáp án: nếu có nút "Trả lời trên lớp" thì mở khung trả lời.
                if safe_is_visible(answer_button):
                    safe_click(answer_button)
                    last_progress = time.monotonic(); stall_reported = False
                    continue
                # Không có gì để làm → chờ ngắn. Nếu kẹt quá lâu, báo lý do 1 lần.
                if not stall_reported and time.monotonic() - last_progress > STALL_SECONDS:
                    log_stall_diagnostics(page, all_buttons)
                    stall_reported = True
                page.wait_for_timeout(400)
                continue

            # Có đáp án → đang tiến triển, reset watchdog.
            last_progress = time.monotonic(); stall_reported = False

            answer_count = answers_locator.count()
            if answer_count == 0:
                continue

            # Khoá ghi nhớ: ưu tiên text câu hỏi, fallback "vân tay" đáp án khi
            # selector text câu hỏi không khớp (nguyên nhân hay gây kẹt im lặng).
            question_text = ""
            if safe_is_visible(question_locator):
                try:
                    question_text = question_locator.inner_text().strip()
                except PlaywrightError:
                    question_text = ""
            if not question_text:
                question_text = answers_fingerprint(answers_locator) or "?"
            print(f"\n[Q] {question_text[:60]}...")

            tried_indices = wrong_answers.get(question_text, set())
            # Reset if we tried all
            if len(tried_indices) >= answer_count:
                tried_indices.clear()

            next_index = next((i for i in range(answer_count) if i not in tried_indices), 0)

            print(f"[Pick] #{next_index + 1}/{answer_count}")
            if not safe_click(answers_locator.nth(next_index)):
                continue  # đáp án chưa render xong vì lag → thử lại vòng sau
            if not safe_click(check_button):
                continue

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

                if safe_is_visible(next_page_button):
                    safe_click(next_page_button)
                    print("[Done] Next Page")
                    try: next_page_button.wait_for(state="hidden", timeout=1000)
                    except PlaywrightError: pass
                elif safe_is_visible(next_button):
                    safe_click(next_button)
                    print("[Done] Next Question")
                    try: next_button.wait_for(state="hidden", timeout=1000)
                    except PlaywrightError: pass
                elif safe_is_visible(retry_button):
                    wrong_answers.setdefault(question_text, set()).add(next_index)
                    print(f"[Wrong] Index {next_index + 1} marked")
                    safe_click(retry_button)
                    try: retry_button.wait_for(state="hidden", timeout=1000)
                    except PlaywrightError: pass
            except PlaywrightTimeoutError:
                # Phản hồi tới chậm (lag) hoặc chưa có nút tiếp theo → vòng sau xử lý lại.
                print("[WARN] Chưa thấy nút tiếp theo (có thể do lag), thử lại...")
            except PlaywrightError:
                print("[WARN] No follow-up button")

        except PlaywrightError as e:
            # Lỗi tạm thời ở bất kỳ đâu trong vòng lặp: nghỉ ngắn rồi tiếp tục.
            if page.is_closed():
                break
            print(f"[WARN] Lỗi tạm thời, tự hồi phục: {str(e)[:80]}")
            try: page.wait_for_timeout(TRANSIENT_BACKOFF_MS)
            except PlaywrightError: break

    try:
        page.wait_for_event("close")
    except PlaywrightError:
        pass
