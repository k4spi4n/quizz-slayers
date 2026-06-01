import json
import os
import re
import sys
import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox
from typing import Optional
from PIL import Image, ImageTk

import pyperclip

from playwright.sync_api import Page

# Add project root to sys.path to import shared modules
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.core.auth import ensure_login_gui

LOGIN_URL = "https://edux.cmcu.edu.vn/login"
PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "questions_prompt.txt")

QUESTION_LABEL_RE = re.compile(r"^Câu\s+(\d+)")
ANSWER_LINE_RE = re.compile(r"^(\d+)\.(.*)$")
TF_TOKEN_RE = re.compile(r"(\d+)\s*\.\s*(đúng|sai|true|false|d|đ|s|t|f|1|0)", re.IGNORECASE)


def ensure_prompt_file() -> None:
    os.makedirs(os.path.dirname(PROMPT_PATH), exist_ok=True)
    if not os.path.exists(PROMPT_PATH):
        with open(PROMPT_PATH, "w", encoding="utf-8") as handle:
            handle.write("")


def load_answers_from_jsonl_line(raw_line: str) -> dict[int, str]:
    line = raw_line.strip()
    if line.startswith("\ufeff"):
        line = line.lstrip("\ufeff").lstrip()

    if line.startswith("[") or line.startswith("{"):
        try:
            data = json.loads(line)
        except Exception:
            data = None
        if data is not None:
            if isinstance(data, dict) and "answers" in data:
                data = data["answers"]
            return normalize_answers_payload(data)

    # Try to split concatenated JSON objects: }{ -> }\n{
    normalized = re.sub(r"}\s*{", "}\n{", line)
    items = []
    for chunk in normalized.splitlines():
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            items.append(json.loads(chunk))
        except Exception:
            continue

    return normalize_answers_payload(items)


def normalize_answers_payload(data) -> dict[int, str]:
    answers: dict[int, str] = {}
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            idx = item.get("so_cau") or item.get("soCau") or item.get("question")
            ans = item.get("dap_an") or item.get("dapAn") or item.get("answer")
            if idx is None or ans is None:
                continue
            try:
                idx_int = int(idx)
            except Exception:
                continue
            if isinstance(ans, list):
                answers[idx_int] = ", ".join(str(x) for x in ans)
            else:
                answers[idx_int] = str(ans).strip()
    elif isinstance(data, dict):
        for key, value in data.items():
            try:
                idx_int = int(key)
            except Exception:
                continue
            if isinstance(value, list):
                answers[idx_int] = ", ".join(str(x) for x in value)
            else:
                answers[idx_int] = str(value).strip()
    return answers


def parse_question_index(text: str) -> Optional[int]:
    match = QUESTION_LABEL_RE.match(text.strip())
    if not match:
        return None
    return int(match.group(1))


def extract_options(options_locator) -> list[dict[str, str]]:
    return options_locator.evaluate_all(
        """
        nodes => nodes.map(node => {
          const letter = node.querySelector('span.flex-shrink-0')?.innerText?.trim() || '';
          const text = node.querySelector('div.prose p')?.innerText?.trim() || '';
          return { letter, text };
        })
        """
    )


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def parse_true_false_answers(answer_value: str, expected_count: int) -> list[bool]:
    normalized = normalize_text(answer_value)
    pairs = TF_TOKEN_RE.findall(normalized)
    if pairs:
        result: list[bool] = []
        for _, token in pairs:
            result.append(token in {"đúng", "d", "đ", "true", "t", "1"})
        return result

    tokens = re.findall(r"[a-zà-ỹ]+|\d", normalized)
    result = []
    for token in tokens:
        if token in {"đúng", "d", "đ", "true", "t", "1"}:
            result.append(True)
        elif token in {"sai", "s", "false", "f", "0"}:
            result.append(False)
        if len(result) >= expected_count:
            break
    return result


def build_compact_prompt_payload(payload_json: dict) -> dict:
    data = payload_json.get("data", {}) if isinstance(payload_json, dict) else {}
    exam_data = data.get("exam_data", {}) if isinstance(data, dict) else {}

    compact: dict[str, object] = {
        "title": data.get("title"),
        "total_questions": data.get("total_questions"),
        "multiple_choice": [],
        "fill_in_blank": [],
        "essay": [],
        "true_false": [],
    }

    for item in exam_data.get("multiple_choice", []) or []:
        compact["multiple_choice"].append(
            {
                "id": item.get("id"),
                "question": item.get("question"),
                "options": item.get("options"),
            }
        )

    for item in exam_data.get("fill_in_blank", []) or []:
        compact["fill_in_blank"].append(
            {
                "id": item.get("id"),
                "question": item.get("question"),
            }
        )

    for item in exam_data.get("essay", []) or []:
        compact["essay"].append(
            {
                "id": item.get("id"),
                "question": item.get("question"),
            }
        )

    for item in exam_data.get("true_false", []) or []:
        statements = [s.get("text") for s in item.get("statements", []) or []]
        compact["true_false"].append(
            {
                "id": item.get("id"),
                "question": item.get("question"),
                "statements": statements,
            }
        )

    return compact


def prompt_answers_via_gui() -> dict[int, str]:
    root = tk.Tk()
    root.title("Paste Answers")
    root.geometry("820x620")
    root.minsize(720, 520)

    ui_font = tkfont.Font(family="Segoe UI", size=10)

    prompt_text = ""
    if os.path.exists(PROMPT_PATH):
        try:
            with open(PROMPT_PATH, "r", encoding="utf-8") as handle:
                prompt_text = handle.read().strip()
        except Exception:
            prompt_text = ""

    top_frame = tk.Frame(root)
    top_frame.pack(fill="both", expand=True, padx=12, pady=(12, 6))

    top_label = tk.Label(
        top_frame,
        text="Danh sách câu hỏi + prompt (đã tạo sẵn)",
        anchor="w",
        justify="left",
        font=ui_font,
    )
    top_label.pack(fill="x", pady=(0, 4))

    top_buttons = tk.Frame(top_frame)
    top_buttons.pack(fill="x", pady=(0, 6))

    copy_button = tk.Button(
        top_buttons,
        text="Copy Prompt",
        command=lambda: pyperclip.copy(prompt_text) if prompt_text else None,
        font=ui_font,
    )
    copy_button.pack(side="left")

    guide_label = tk.Label(
        top_buttons,
        text="Copy prompt để dán vào bất kỳ Chatbot AI."
             " Sau đó dán kết quả trả về vào khung bên dưới.",
        font=("Segoe UI", 9),
        fg="gray",
    )
    guide_label.pack(side="left", padx=10)

    prompt_box = tk.Text(top_frame, wrap="word", font=ui_font, height=8)
    prompt_box.pack(fill="both", expand=True)
    prompt_box.insert("1.0", prompt_text)
    prompt_box.configure(state="disabled")

    bottom_frame = tk.Frame(root)
    bottom_frame.pack(fill="both", expand=True, padx=12, pady=(6, 12))

    instruction = (
        "Paste one-line JSONL or one-line JSON array below, then click Validate."
    )
    label = tk.Label(bottom_frame, text=instruction, anchor="w", justify="left", font=ui_font)
    label.pack(fill="x", pady=(0, 4))

    text = tk.Text(bottom_frame, wrap="word", font=ui_font)
    text.pack(fill="both", expand=True, pady=8)

    result: dict[int, str] = {}

    def on_validate() -> None:
        raw = text.get("1.0", "end").strip()
        if not raw:
            messagebox.showwarning("Invalid", "Input is empty.")
            return
        answers = load_answers_from_jsonl_line(raw)
        if not answers:
            messagebox.showerror("Invalid", "Could not parse valid answers.")
            return
        result.update(answers)
        root.destroy()

    def on_paste() -> None:
        try:
            clip = root.clipboard_get()
        except Exception:
            messagebox.showerror("Clipboard", "Clipboard is empty or unavailable.")
            return
        text.delete("1.0", "end")
        text.insert("1.0", clip)

    buttons = tk.Frame(bottom_frame)
    buttons.pack(fill="x", pady=(0, 12))

    paste_button = tk.Button(buttons, text="Paste from Clipboard", command=on_paste, font=ui_font)
    paste_button.pack(side="left", padx=6)

    validate_button = tk.Button(
        buttons,
        text="Validate",
        command=on_validate,
        font=("Segoe UI", 10, "bold"),
        bg="#4CAF50",
        fg="white",
    )
    validate_button.pack(side="left", padx=6)

    root.update_idletasks()
    req_w = max(root.winfo_reqwidth() + 40, 820)
    req_h = max(root.winfo_reqheight() + 80, 620)
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    margin = 20
    width = min(req_w, screen_w - 40)
    height = min(req_h, screen_h - 80)
    x = screen_w - width - margin
    y = screen_h - height - margin - 40
    root.geometry(f"{width}x{height}+{x}+{y}")

    root.mainloop()
    return result


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


def get_answers_via_litellm(prompt_content: str, api_key: str, model: str) -> dict[int, str]:
    if not api_key:
        return {}
    try:
        from litellm import completion
    except Exception as exc:
        print(f"[WARN] LiteLLM not available: {exc}")
        return {}

    model_name = resolve_litellm_model(model, api_key)
    try:
        response = completion(
            model=model_name,
            messages=[{"role": "user", "content": prompt_content}],
            api_key=api_key,
            temperature=0,
        )
    except Exception as exc:
        print(f"[WARN] LiteLLM request failed: {exc}")
        return {}

    content = ""
    try:
        content = response["choices"][0]["message"]["content"]
    except Exception:
        try:
            content = response.choices[0].message.content
        except Exception:
            content = ""

    content = sanitize_ai_response(content)
    return load_answers_from_jsonl_line(content)



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


def test_bruteforce(page: Page) -> None:
    ensure_prompt_file()
    email, password, api_key, model = ensure_login_gui()

    page.goto(LOGIN_URL, wait_until="domcontentloaded")
    
    if email and password:
        page.locator("#email").fill(email)
        page.locator("#password").fill(password)
        page.locator("#password").press("Enter")
        print("\n[INFO] Auto-login attempted. Finish navigation to the test.")
    else:
        print("\n[INFO] 'Tự đăng nhập' được chọn. Vui lòng đăng nhập thủ công trên trình duyệt.")

    show_start_dialog("Khi bạn thấy màn hình chuẩn bị làm bài tập, hãy nhấn nút dưới đây để bắt đầu.")

    start_button = page.get_by_role("button", name="Làm bài tập")

    with page.expect_response(lambda resp: "start" in resp.url, timeout=20000) as response_info:
        start_button.click()

    response = response_info.value
    payload_text = ""
    try:
        payload_json = response.json()
        compact_payload = build_compact_prompt_payload(payload_json)
        payload_text = json.dumps(compact_payload, ensure_ascii=False, indent=2)
    except Exception:
        payload_text = response.text()

    prompt_line = (
        "trả về JSONL một dòng duy nhất (không xuống dòng); "
        "mỗi phần tử có so_cau và dap_an; "
        "dap_an là A/B/C/D hoặc từ/cụm từ/văn bản cần điền; "
        "với câu đúng/sai, dap_an là mảng giá trị Đúng/Sai theo thứ tự mệnh đề; "
        "không giải thích gì thêm"
    )
    prompt_content = payload_text.strip() + "\n\n" + prompt_line + "\n"
    with open(PROMPT_PATH, "w", encoding="utf-8") as handle:
        handle.write(prompt_content)

    try:
        pyperclip.copy(prompt_content)
        print("[INFO] Prompt copied to clipboard.")
    except Exception as exc:
        print(f"[WARN] Clipboard copy failed: {exc}")

    print("[INFO] Wrote questions prompt to questions_prompt.txt.")

    answers = {}
    if api_key:
        print("[INFO] API key detected. Requesting answers via LiteLLM...")
        answers = get_answers_via_litellm(prompt_content, api_key, model)
        if not answers:
            print("[WARN] LiteLLM did not return usable answers. Falling back to manual paste.")

    if not answers:
        print("[INFO] Paste the prompt into AI, then paste answers into the dialog.")
        answers = prompt_answers_via_gui()
    if not answers:
        print("[WARN] No answers parsed from dialog input. Stopping.")
        return

    dialog = page.locator("div[role='dialog'][data-slot='dialog-content']")
    options_locator = dialog.locator(
        "div.relative.flex.items-center.space-x-2.p-2.border.rounded-lg.cursor-pointer"
    )
    input_locator = dialog.locator("input[type='text']")
    textarea_locator = dialog.locator("textarea")
    true_false_blocks = dialog.locator("div.border.border-gray-200.rounded-lg.p-3.bg-gray-50")
    next_button = page.get_by_role("button", name="Câu tiếp")
    submit_button = page.get_by_role("button", name="Nộp bài")

    while not page.is_closed():
        try:
            dialog.wait_for(state="visible", timeout=10000)
            label_handle = page.wait_for_function(
                """
                () => {
                  const dialog = document.querySelector("div[role='dialog'][data-slot='dialog-content']");
                  if (!dialog) return null;
                  const label = Array.from(dialog.querySelectorAll('span'))
                    .find(s => (s.textContent || '').trim().startsWith('Câu '));
                  return label ? label.textContent.trim() : null;
                }
                """,
                timeout=10000,
            )
            label_text = label_handle.json_value()
        except Exception:
            print("[WARN] Question label not visible yet. Retrying.")
            page.wait_for_timeout(200)
            continue

        question_index = parse_question_index(label_text)
        if question_index is None:
            print(f"[WARN] Could not parse question index from: {label_text}")
            page.wait_for_timeout(200)
            continue

        answer_value = answers.get(question_index, "").strip()
        if not answer_value:
            print(f"[WARN] No answer for question {question_index}. Skipping.")
        else:
            if true_false_blocks.first.is_visible():
                blocks_count = true_false_blocks.count()
                tf_answers = parse_true_false_answers(answer_value, blocks_count)
                if len(tf_answers) < blocks_count:
                    print("[WARN] Not enough true/false answers to fill.")
                else:
                    print(f"[INFO] Question {question_index}: filling true/false")
                    for i in range(blocks_count):
                        block = true_false_blocks.nth(i)
                        if tf_answers[i]:
                            block.get_by_role("button", name="Đúng").click()
                        else:
                            block.get_by_role("button", name="Sai").click()
            elif textarea_locator.is_visible():
                print(f"[INFO] Question {question_index}: filling textarea")
                textarea_locator.fill(answer_value)
            elif input_locator.is_visible():
                print(f"[INFO] Question {question_index}: filling input")
                input_locator.fill(answer_value)
            else:
                try:
                    options_locator.first.wait_for(state="visible", timeout=10000)
                except Exception:
                    print("[WARN] Options not visible yet. Retrying.")
                    page.wait_for_timeout(200)
                    continue

                print(f"[INFO] Question {question_index}: selecting {answer_value}")
                options = extract_options(options_locator)
                chosen_index = None

                if len(answer_value) == 1 and answer_value.upper() in {"A", "B", "C", "D"}:
                    target_letter = f"{answer_value.upper()}."
                    for i, option in enumerate(options):
                        if option["letter"].startswith(target_letter):
                            chosen_index = i
                            break
                else:
                    target = normalize_text(answer_value)
                    for i, option in enumerate(options):
                        option_text = normalize_text(option["text"])
                        if target and target in option_text:
                            chosen_index = i
                            break

                if chosen_index is None:
                    print("[WARN] No matching option found.")
                else:
                    options_locator.nth(chosen_index).click()

        if submit_button.is_visible():
            submit_button.click()
            print("[INFO] Clicked 'Nop bai'.")
            break

        if next_button.is_visible():
            current_label = label_text
            current_progress = dialog.evaluate(
                """
                (node) => {
                  const progress = node.querySelector('span.text-gray-700');
                  return progress ? progress.textContent.trim() : '';
                }
                """
            )
            next_button.click()
            new_label = dialog.evaluate(
                """
                (node) => {
                  const label = Array.from(node.querySelectorAll('span'))
                    .find(s => (s.textContent || '').trim().startsWith('Câu '));
                  return label ? label.textContent.trim() : '';
                }
                """
            )
            if new_label and new_label != current_label:
                continue
            try:
                page.wait_for_function(
                    """
                    (prevLabel, prevProgress) => {
                      const dialog = document.querySelector("div[role='dialog'][data-slot='dialog-content']");
                      if (!dialog) return false;
                      const label = Array.from(dialog.querySelectorAll('span'))
                        .find(s => (s.textContent || '').trim().startsWith('Câu '));
                      const progress = dialog.querySelector('span.text-gray-700');
                      const labelChanged = label && label.textContent.trim() !== prevLabel;
                      const progressChanged = progress && progress.textContent.trim() !== prevProgress;
                      return labelChanged || progressChanged;
                    }
                    """,
                    current_label,
                    current_progress,
                    timeout=10000,
                )
            except Exception:
                print("[WARN] Next question did not appear yet.")
        else:
            page.wait_for_timeout(200)

    print("[INFO] Browser will stay open. Close the browser window to finish.")
    page.wait_for_event("close", timeout=0)
