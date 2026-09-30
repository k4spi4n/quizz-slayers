import os
import random
import re
import time
import tkinter as tk
import sys
from typing import Optional
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


def safe_is_enabled(locator) -> bool:
    """is_visible() VÀ is_enabled() đồng thời nuốt lỗi tạm thời."""
    try:
        if not locator.is_visible(timeout=500):
            return False
        if not locator.is_enabled(timeout=500):
            return False
        aria_disabled = locator.get_attribute("aria-disabled", timeout=500)
        if aria_disabled == "true":
            return False
        class_name = locator.get_attribute("class", timeout=500) or ""
        if "cursor-not-allowed" in class_name and "pointer-events-none" in class_name:
            return False
        return True
    except PlaywrightError:
        return False


def safe_click(locator, timeout: int = ACTION_TIMEOUT_MS) -> bool:
    """Click có auto-wait + nuốt lỗi tạm thời. Trả về True nếu click thành công."""
    try:
        locator.click(timeout=timeout)
        return True
    except PlaywrightError:
        try:
            locator.click(force=True, timeout=1000)
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


def get_active_dialog(page: Page):
    """Lấy container của modal dialog nếu đang hiển thị trên màn hình."""
    dialog_selectors = [
        "div[role='dialog']",
        "div.sm\\:max-w-\\[100\\%\\]",
        "[aria-modal='true']",
        "div[data-state='open'][role='dialog']",
    ]
    for sel in dialog_selectors:
        try:
            loc = page.locator(sel)
            if loc.count() > 0 and safe_is_visible(loc.first):
                return loc.first
        except PlaywrightError:
            pass
    return None


def get_dialog_next_page_button(page: Page):
    """Tìm nút 'Trang sau' bên trong popup quiz (thường có nền xanh lá bg-green-600)."""
    selectors = [
        "button.bg-green-600:has-text('Trang sau')",
        "button[class*='bg-green']:has-text('Trang sau')",
        "div[role='dialog'] button.bg-green-600",
        "div.sm\\:max-w-\\[100\\%\\] button.bg-green-600",
        "div[role='dialog'] button:has-text('Trang sau')",
        "div.sm\\:max-w-\\[100\\%\\] button:has-text('Trang sau')",
        "div[role='dialog'] button:has-text('Tiếp tục')",
        "div[role='dialog'] button:has-text('Hoàn thành')",
    ]
    for sel in selectors:
        try:
            loc = page.locator(sel)
            for i in range(loc.count()):
                btn = loc.nth(i)
                txt = (btn.inner_text() or "").strip()
                if any(bad in txt for bad in ["Bài giảng", "Khóa học", "Thử lại", "Bỏ qua", "Phản hồi", "Đổi câu hỏi"]):
                    continue
                if safe_is_enabled(btn):
                    return btn
        except PlaywrightError:
            pass

    dialog = get_active_dialog(page)
    if dialog:
        for text in ["Trang sau", "Tiếp tục", "Hoàn thành"]:
            try:
                loc = dialog.locator(f"button:has-text('{text}')")
                for i in range(loc.count()):
                    btn = loc.nth(i)
                    txt = (btn.inner_text() or "").strip()
                    if "Bài giảng" in txt:
                        continue
                    if safe_is_enabled(btn):
                        return btn
            except PlaywrightError:
                pass

    return None


def get_dialog_retry_button(page: Page):
    """Tìm nút 'Thử lại' (trong dialog hoặc trên thanh công cụ/trên trang)."""
    selectors = [
        "div[role='dialog'] button:has-text('Thử lại')",
        "button:has-text('Thử lại')",
        "button[title*='Thử lại']",
        "[role='button']:has-text('Thử lại')",
        "div.cursor-pointer:has-text('Thử lại')",
        "div:has-text('Thử lại')",
    ]
    for sel in selectors:
        try:
            loc = page.locator(sel)
            for i in range(loc.count()):
                btn = loc.nth(i)
                txt = (btn.inner_text() or "").strip()
                if txt in ["Thử lại", "Thử lại câu hỏi"] or len(txt) <= 25:
                    if safe_is_enabled(btn):
                        return btn
        except PlaywrightError:
            pass
    return None


def get_dialog_skip_button(page: Page):
    """Tìm nút 'Bỏ qua' (đếm ngược tự động chuyển)."""
    selectors = [
        "button:has-text('Bỏ qua')",
        "button[title*='Bỏ qua']",
    ]
    for sel in selectors:
        try:
            loc = page.locator(sel)
            for i in range(loc.count()):
                btn = loc.nth(i)
                if safe_is_enabled(btn):
                    return btn
        except PlaywrightError:
            pass
    return None


def get_dialog_next_question_button(page: Page):
    """Tìm nút 'Câu tiếp theo'."""
    selectors = [
        "div[role='dialog'] button:has-text('Câu tiếp theo')",
        "div.sm\\:max-w-\\[100\\%\\] button:has-text('Câu tiếp theo')",
        "button:has-text('Câu tiếp theo')",
        "button[title*='Câu tiếp theo']",
    ]
    for sel in selectors:
        try:
            loc = page.locator(sel)
            for i in range(loc.count()):
                btn = loc.nth(i)
                if safe_is_enabled(btn):
                    return btn
        except PlaywrightError:
            pass
    return None


def find_action_button(page: Page, names: list[str], must_be_enabled: bool = True):
    """Tìm button theo text, title, aria-label hoặc accessible role name (chịu được chế độ icon thu nhỏ).
    Ưu tiên tìm bên trong dialog trước (nếu có dialog mở), và duyệt qua tất cả elements thay vì chỉ lấy .first."""
    check_fn = safe_is_enabled if must_be_enabled else safe_is_visible

    dialog = get_active_dialog(page)
    containers = [dialog, page] if dialog else [page]

    for container in containers:
        for name in names:
            if name == "Trang sau":
                green_btn = container.locator("button.bg-green-600, button[class*='bg-green']")
                try:
                    for i in range(green_btn.count()):
                        btn = green_btn.nth(i)
                        if check_fn(btn):
                            return btn
                except PlaywrightError:
                    pass

            selectors = [
                f"button:has-text('{name}')",
                f"button[title*='{name}']",
                f"button[aria-label*='{name}']",
            ]
            for sel in selectors:
                loc = container.locator(sel)
                try:
                    for i in range(loc.count()):
                        btn = loc.nth(i)
                        if check_fn(btn):
                            return btn
                except PlaywrightError:
                    pass

            try:
                role_loc = container.get_by_role("button", name=name)
                for i in range(role_loc.count()):
                    btn = role_loc.nth(i)
                    if check_fn(btn):
                        return btn
            except PlaywrightError:
                pass

    return None


def safe_next_slide(page: Page) -> bool:
    """Chuyển sang slide tiếp theo: ưu tiên nút 'Trang sau' trong dialog/slide, fallback phím ArrowRight."""
    dlg_btn = get_dialog_next_page_button(page)
    if dlg_btn and safe_click(dlg_btn):
        return True

    next_btn = find_action_button(page, ["Trang sau"], must_be_enabled=True)
    if next_btn and safe_click(next_btn):
        return True

    try:
        page.keyboard.press("ArrowRight")
        return True
    except PlaywrightError:
        return False


def get_answers_locator(page: Page):
    """Tìm danh sách các lựa chọn đáp án theo nhiều tầng fallback để thích ứng với thay đổi layout."""
    radiogroup_children = page.locator("div[role='radiogroup'] > div")
    if radiogroup_children.count() > 0 and safe_is_visible(radiogroup_children.first):
        return radiogroup_children

    choice_cards = page.locator("div.rounded-xl.border-2").filter(
        has=page.locator("button[role='radio'], span.font-bold")
    )
    if choice_cards.count() > 0 and safe_is_visible(choice_cards.first):
        return choice_cards

    radios = page.locator("button[role='radio']")
    if radios.count() > 0 and safe_is_visible(radios.first):
        return radios

    min_h_cards = page.locator("div.border-2.rounded-xl.min-h-\\[80px\\]")
    if min_h_cards.count() > 0 and safe_is_visible(min_h_cards.first):
        return min_h_cards

    pointer_cards = page.locator("div.border-2.cursor-pointer")
    if pointer_cards.count() > 0 and safe_is_visible(pointer_cards.first):
        return pointer_cards

    return page.locator("div[role='radiogroup'] > div")


def get_question_text(page: Page, answers_loc) -> str:
    """Lấy nội dung câu hỏi một cách linh hoạt, fallback sang vân tay đáp án."""
    candidates = [
        page.locator("div.bg-blue-50.border-blue-500").first,
        page.locator("[class*='text-blue-800']").first,
        page.locator("div.bg-blue-50").first,
        page.locator("p.my-3.text-gray-800.leading-relaxed").first,
        page.locator("div[role='dialog'] h3").first,
        page.locator("div[role='dialog'] .font-semibold").first,
    ]
    for loc in candidates:
        if safe_is_visible(loc):
            try:
                txt = loc.inner_text().strip()
                if txt:
                    return txt
            except PlaywrightError:
                pass

    return answers_fingerprint(answers_loc) or "?"


def extract_revealed_correct_index(page: Page) -> Optional[int]:
    """Khi trả lời sai, EDUX hiển thị 'Đáp án đúng: X.' trên màn hình,
    hoặc viền xanh lá (border-green / bg-green) vào thẻ đáp án đúng.
    Hàm này bóc tách chữ cái hoặc thẻ xanh đó để bot lập tức chọn đúng ngay lần thử tiếp theo."""
    selectors = [
        "div.text-red-700:has-text('Đáp án đúng:')",
        "[class*='text-red']:has-text('Đáp án đúng:')",
        "div:has-text('Đáp án đúng:')",
        "p:has-text('Đáp án đúng:')",
    ]
    for sel in selectors:
        try:
            loc = page.locator(sel)
            if loc.count() > 0 and safe_is_visible(loc.first):
                text = loc.first.inner_text()
                match = re.search(r"Đáp án đúng:\s*([A-Za-z])\b", text)
                if match:
                    letter = match.group(1).upper()
                    return ord(letter) - ord('A')
        except PlaywrightError:
            pass

    # Method 2: Inspect cards for green border/background
    try:
        green_idx = page.evaluate("""
            () => {
                const isGreen = (el) => {
                    if (!el) return false;
                    const cls = el.className || '';
                    if (typeof cls === 'string') {
                        if ((cls.includes('border-green') || cls.includes('bg-green') || cls.includes('border-emerald') || cls.includes('bg-emerald')) &&
                            !cls.includes('border-red') && !cls.includes('bg-red')) {
                            return true;
                        }
                    }
                    try {
                        const style = window.getComputedStyle(el);
                        for (const colorStr of [style.borderColor, style.backgroundColor]) {
                            const m = (colorStr || '').match(/rgba?\\((\\d+),\\s*(\\d+),\\s*(\\d+)/);
                            if (m) {
                                const r = parseInt(m[1], 10), g = parseInt(m[2], 10), b = parseInt(m[3], 10);
                                if (g >= 120 && g > r * 1.25 && g > b * 1.1) return true;
                            }
                        }
                    } catch (e) {}
                    return false;
                };

                const cards = Array.from(document.querySelectorAll("div.rounded-xl.border-2, div[role='radiogroup'] > div, div.border-2.rounded-xl, div.border-2.cursor-pointer"));
                for (let i = 0; i < cards.length; i++) {
                    const c = cards[i];
                    if (isGreen(c) || c.querySelector("[class*='border-green'], [class*='bg-green'], svg.text-green-500")) {
                        const red = c.querySelector("[class*='border-red'], [class*='bg-red']");
                        if (!red) return i;
                    }
                }
                return null;
            }
        """)
        if green_idx is not None and isinstance(green_idx, int):
            return green_idx
    except PlaywrightError:
        pass

    return None


def answers_fingerprint(answers_locator) -> str:
    """Khoá ghi nhớ dựa trên nội dung các đáp án — dùng khi không lấy được text câu hỏi."""
    try:
        texts = answers_locator.all_inner_texts()
        joined = " | ".join(t.strip() for t in texts if t.strip())
        return joined[:200]
    except PlaywrightError:
        return ""


def log_stall_diagnostics(page: Page) -> None:
    """In ra trạng thái màn hình khi nghi bị kẹt, để biết LÝ DO thay vì im lặng."""
    try:
        url = page.url
    except PlaywrightError:
        url = "?"

    dialog_open = get_active_dialog(page) is not None
    visible_buttons = []
    try:
        all_btns = page.locator("button:visible")
        for i in range(min(all_btns.count(), 10)):
            b = all_btns.nth(i)
            txt = (b.inner_text() or "").strip().replace("\n", " ")
            title = b.get_attribute("title") or ""
            enabled = b.is_enabled()
            cls = (b.get_attribute("class") or "")[:25]
            desc = f"'{txt or title}'({'enabled' if enabled else 'disabled'}, cls={cls})"
            visible_buttons.append(desc)
    except PlaywrightError:
        pass

    radios_count = 0
    try:
        radios_count = page.locator(
            "button[role='radio'], div[role='radiogroup'] > div, div.rounded-xl.border-2"
        ).count()
    except PlaywrightError:
        pass

    print(
        f"[STALL DIAG] URL: {url} | Dialog mở: {dialog_open} | "
        f"Cards/Radio: {radios_count} | Buttons: {', '.join(visible_buttons) or 'không có'}"
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
    known_correct_answers: dict[str, int] = {}

    last_progress = time.monotonic()
    stall_reported = False
    STALL_SECONDS = 8.0

    while not page.is_closed():
        # Mỗi vòng lặp được cô lập: lỗi tạm thời (mất mạng, context bị huỷ khi
        # điều hướng, element detach do re-render) chỉ làm bỏ qua 1 vòng rồi thử
        # lại, KHÔNG làm sập cả script.
        try:
            # =============================================================
            # BƯỚC 1: Xử lý các trạng thái hoàn thành / chuyển tiếp ưu tiên cao
            # (Phải kiểm tra TRƯỚC để tránh kẹt khi quiz đã xong mà câu hỏi vẫn còn trên DOM)
            # =============================================================

            # 1.1. Nút "Trang sau" trong Dialog (khi quiz đã hoàn thành, xuất hiện nút xanh lá):
            dialog_next_btn = get_dialog_next_page_button(page)
            if dialog_next_btn:
                print("[Done] Phát hiện nút 'Trang sau' trong popup quiz, đang chuyển slide...")
                safe_click(dialog_next_btn)
                try:
                    dialog_next_btn.wait_for(state="hidden", timeout=3000)
                except PlaywrightError:
                    pass
                page.wait_for_timeout(300)
                last_progress = time.monotonic()
                stall_reported = False
                continue

            # 1.2. Nút "Bỏ qua" đếm ngược (khi trả lời đúng và EDUX đếm ngược 3-5s):
            skip_btn = get_dialog_skip_button(page)
            if skip_btn:
                print("[Done] Bấm nút 'Bỏ qua' (Skip Countdown)...")
                safe_click(skip_btn)
                try:
                    skip_btn.wait_for(state="hidden", timeout=1500)
                except PlaywrightError:
                    pass
                page.wait_for_timeout(200)
                d_next = get_dialog_next_page_button(page)
                if d_next:
                    print("[Done] Bấm tiếp 'Trang sau' sau khi bỏ qua...")
                    safe_click(d_next)
                    try:
                        d_next.wait_for(state="hidden", timeout=3000)
                    except PlaywrightError:
                        pass
                last_progress = time.monotonic()
                stall_reported = False
                continue

            # 1.3. Nút "Câu tiếp theo" (bài quiz có nhiều câu hỏi):
            next_q_btn = get_dialog_next_question_button(page)
            if next_q_btn:
                print("[Done] Chuyển 'Câu tiếp theo'...")
                safe_click(next_q_btn)
                try:
                    next_q_btn.wait_for(state="hidden", timeout=3000)
                except PlaywrightError:
                    pass
                page.wait_for_timeout(300)
                last_progress = time.monotonic()
                stall_reported = False
                continue

            # 1.4. Nút "Thử lại" (khi trả lời sai):
            retry_btn = get_dialog_retry_button(page)
            if retry_btn:
                print("[INFO] Phát hiện nút 'Thử lại', chuẩn bị thử lại câu hỏi...")
                revealed_idx = extract_revealed_correct_index(page)
                if revealed_idx is not None:
                    q_text = get_question_text(page, get_answers_locator(page))
                    known_correct_answers[q_text] = revealed_idx
                    print(f"[Revealed] Ghi nhớ đáp án đúng: #{revealed_idx + 1}")
                safe_click(retry_btn)
                try:
                    retry_btn.wait_for(state="hidden", timeout=5000)
                except PlaywrightError:
                    pass
                page.wait_for_timeout(300)
                last_progress = time.monotonic()
                stall_reported = False
                continue

            # 1.5. Slide không có câu hỏi -> sang trang kế tiếp:
            no_question_btn = find_action_button(page, ["Không có câu hỏi"], must_be_enabled=False)
            if no_question_btn:
                print("[INFO] Slide không có câu hỏi -> Chuyển slide tiếp theo")
                safe_next_slide(page)
                try:
                    no_question_btn.wait_for(state="hidden", timeout=1500)
                except PlaywrightError:
                    pass
                page.wait_for_timeout(300)
                last_progress = time.monotonic()
                stall_reported = False
                continue

            # =============================================================
            # BƯỚC 2: Kiểm tra trạng thái slide / mở popup câu hỏi
            # =============================================================
            answers_locator = get_answers_locator(page)
            answer_count = answers_locator.count()
            answers_visible = answer_count > 0 and safe_is_visible(answers_locator.first)

            if not answers_visible:
                # 2.1. Mở popup câu hỏi nếu có nút "Trả lời trên lớp" hoặc "Hỏi trên lớp"
                answer_button = find_action_button(page, ["Trả lời trên lớp", "Hỏi trên lớp"])
                if answer_button:
                    print("[INFO] Bấm mở popup câu hỏi...")
                    safe_click(answer_button)
                    page.wait_for_timeout(500)
                    last_progress = time.monotonic()
                    stall_reported = False
                    continue

                # 2.2. Nếu slide đang tải câu hỏi ("Đang kiểm tra...")
                if page.locator("text='Đang kiểm tra...'").count() > 0:
                    page.wait_for_timeout(500)
                    continue

                # 2.3. Slide đã hoàn thành hoặc không có câu hỏi: nút 'Trang sau' ở slide bar đang ENABLED
                if not get_active_dialog(page):
                    slide_next_btn = find_action_button(page, ["Trang sau"], must_be_enabled=True)
                    if slide_next_btn:
                        print("[INFO] Bấm 'Trang sau' trên thanh điều khiển slide...")
                        safe_click(slide_next_btn)
                        page.wait_for_timeout(500)
                        last_progress = time.monotonic()
                        stall_reported = False
                        continue

                # 2.4. Không có gì để làm -> chờ ngắn. Nếu kẹt quá lâu, tự gỡ kẹt bằng phím ArrowRight
                if not stall_reported and time.monotonic() - last_progress > STALL_SECONDS:
                    log_stall_diagnostics(page)
                    if not get_active_dialog(page):
                        print("[RECOVERY] Thử nhấn phím ArrowRight để chuyển slide...")
                        page.keyboard.press("ArrowRight")
                    stall_reported = True
                    last_progress = time.monotonic()

                page.wait_for_timeout(400)
                continue

            # =============================================================
            # BƯỚC 3: Trả lời câu hỏi (Đang có danh sách đáp án hiển thị)
            # =============================================================
            last_progress = time.monotonic()
            stall_reported = False

            question_text = get_question_text(page, answers_locator)
            print(f"\n[Q] {question_text[:60]}...")

            if question_text in known_correct_answers and known_correct_answers[question_text] < answer_count:
                next_index = known_correct_answers[question_text]
                print(f"[Pick Known Correct] #{next_index + 1}/{answer_count}")
            else:
                tried_indices = wrong_answers.get(question_text, set())
                if len(tried_indices) >= answer_count:
                    tried_indices.clear()
                next_index = next((i for i in range(answer_count) if i not in tried_indices), 0)
                print(f"[Pick] #{next_index + 1}/{answer_count}")

            # Chọn đáp án
            option_card = answers_locator.nth(next_index)
            radio_inside = option_card.locator("button[role='radio']")
            clicked = False
            if radio_inside.count() > 0 and safe_is_visible(radio_inside.first):
                clicked = safe_click(radio_inside.first)
            if not clicked:
                clicked = safe_click(option_card)

            if not clicked:
                continue

            page.wait_for_timeout(200)

            # Chờ nút "Kiểm tra" trở thành enabled sau khi chọn đáp án (nếu có)
            instant_action = get_dialog_retry_button(page) or get_dialog_next_question_button(page) or get_dialog_skip_button(page)
            check_button = None
            if not instant_action:
                for _ in range(5):
                    check_button = find_action_button(page, ["Kiểm tra"], must_be_enabled=True)
                    if check_button:
                        break
                    instant_action = get_dialog_retry_button(page) or get_dialog_next_question_button(page) or get_dialog_skip_button(page)
                    if instant_action:
                        break
                    page.wait_for_timeout(200)

            if check_button and safe_is_enabled(check_button):
                safe_click(check_button)
            elif not instant_action:
                print("[INFO] Trắc nghiệm nộp tức thì, đang chờ kết quả...")

            # =============================================================
            # BƯỚC 4: Xử lý ngay kết quả sau khi nộp
            # =============================================================
            try:
                page.wait_for_function(
                    """
                    () => {
                      const targets = ['Thử lại', 'Bỏ qua', 'Câu tiếp theo', 'Trang sau'];
                      const elements = Array.from(document.querySelectorAll("button, a[role='button'], div[role='button'], div.cursor-pointer, [role='button']"));
                      return elements.some(b => {
                        const txt = (b.textContent || '').trim();
                        const title = b.getAttribute('title') || '';
                        return targets.some(t => txt.includes(t) || title.includes(t)) && !b.disabled && b.offsetParent !== null;
                      });
                    }
                    """,
                    timeout=10000,
                )

                # 4.1. ƯU TIÊN 1: Nếu trả lời SAI -> nút 'Thử lại' xuất hiện
                retry_button = get_dialog_retry_button(page)
                if retry_button:
                    revealed_idx = extract_revealed_correct_index(page)
                    if revealed_idx is not None and revealed_idx < answer_count:
                        known_correct_answers[question_text] = revealed_idx
                        print(f"[Revealed] Đáp án đúng được hiển thị: #{revealed_idx + 1}")
                    else:
                        wrong_answers.setdefault(question_text, set()).add(next_index)
                        print(f"[Wrong] Index {next_index + 1} marked")

                    safe_click(retry_button)
                    print("[Retry] Clicked 'Thử lại'")
                    try:
                        retry_button.wait_for(state="hidden", timeout=5000)
                    except PlaywrightError:
                        pass
                    last_progress = time.monotonic()
                    stall_reported = False
                    continue

                # 4.2. ƯU TIÊN 2: Nút "Bỏ qua" đếm ngược
                skip_btn = get_dialog_skip_button(page)
                if skip_btn:
                    safe_click(skip_btn)
                    print("[Done] Clicked 'Bỏ qua' (Skip Countdown)")
                    try:
                        skip_btn.wait_for(state="hidden", timeout=1500)
                    except PlaywrightError:
                        pass

                # 4.3. ƯU TIÊN 3: Nút "Câu tiếp theo"
                next_button = get_dialog_next_question_button(page)
                if next_button:
                    safe_click(next_button)
                    print("[Done] Next Question")
                    try:
                        next_button.wait_for(state="hidden", timeout=2000)
                    except PlaywrightError:
                        pass
                    last_progress = time.monotonic()
                    stall_reported = False
                    continue

                # 4.4. ƯU TIÊN 4: Nút "Trang sau" trong Dialog
                dialog_next = get_dialog_next_page_button(page)
                if dialog_next:
                    safe_click(dialog_next)
                    print("[Done] Clicked 'Trang sau' in Dialog")
                    try:
                        dialog_next.wait_for(state="hidden", timeout=3000)
                    except PlaywrightError:
                        pass
                    page.wait_for_timeout(300)
                    last_progress = time.monotonic()
                    stall_reported = False
                    continue

                # 4.5. Nút "Trang sau" trên Slide bar (nếu dialog đã tự đóng)
                slide_next = find_action_button(page, ["Trang sau"], must_be_enabled=True)
                if slide_next:
                    safe_click(slide_next)
                    print("[Done] Clicked 'Trang sau' on Slide")
                    try:
                        slide_next.wait_for(state="hidden", timeout=2000)
                    except PlaywrightError:
                        pass
                    last_progress = time.monotonic()
                    stall_reported = False
                    continue

            except PlaywrightTimeoutError:
                # Phản hồi tới chậm (lag) hoặc chưa có nút tiếp theo -> vòng sau xử lý lại.
                print("[WARN] Chưa thấy nút phản hồi sau 'Kiểm tra' (có thể do lag), vòng sau sẽ tự kiểm tra lại...")
            except PlaywrightError:
                print("[WARN] Lỗi khi xử lý nút phản hồi")

        except PlaywrightError as e:
            # Lỗi tạm thời ở bất kỳ đâu trong vòng lặp: nghỉ ngắn rồi tiếp tục.
            if page.is_closed():
                break
            print(f"[WARN] Lỗi tạm thời, tự hồi phục: {str(e)[:80]}")
            try:
                page.wait_for_timeout(TRANSIENT_BACKOFF_MS)
            except PlaywrightError:
                break

    try:
        page.wait_for_event("close")
    except PlaywrightError:
        pass
