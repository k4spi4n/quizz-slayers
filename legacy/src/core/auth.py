import os
import tkinter as tk
from tkinter import messagebox
import sys


def get_root_dir():
    # Attempt to find the root directory of the project
    current = os.path.dirname(os.path.abspath(__file__))
    # We are in src/core, so root should be 2 levels up
    # src/core -> src -> root
    root = os.path.dirname(os.path.dirname(current))
    if os.path.exists(os.path.join(root, "run_test_solver.bat")):
        return root

    # Fallback search for .git or known root files
    temp = current
    while temp != os.path.dirname(temp):
        if os.path.exists(os.path.join(temp, ".git")) or \
           os.path.exists(os.path.join(temp, "README.md")):
            return temp
        temp = os.path.dirname(temp)
    return root


ROOT_DIR = get_root_dir()
ENV_PATH = os.path.join(ROOT_DIR, ".env")


def is_legacy_env_file() -> bool:
    if not os.path.exists(ENV_PATH):
        return False
    has_email = False
    has_password = False
    has_api_key = False
    has_model = False
    with open(ENV_PATH, "r", encoding="utf-8") as env_file:
        for line in env_file:
            raw = line.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            key = raw.split("=", 1)[0]
            if key == "EDUX_EMAIL":
                has_email = True
            elif key == "EDUX_PASSWORD":
                has_password = True
            elif key == "EDUX_API_KEY":
                has_api_key = True
            elif key == "EDUX_MODEL":
                has_model = True
    return (has_email and has_password) and not (has_api_key or has_model)


def load_env_file() -> None:
    if not os.path.exists(ENV_PATH):
        return
    with open(ENV_PATH, "r", encoding="utf-8") as env_file:
        for line in env_file:
            raw = line.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            if key and key not in os.environ:
                os.environ[key] = value


def save_env_file(email: str, password: str, api_key: str, model: str) -> None:
    env_data = {}
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                env_data[k] = v

    env_data["EDUX_EMAIL"] = email
    env_data["EDUX_PASSWORD"] = password
    env_data["EDUX_API_KEY"] = api_key
    env_data["EDUX_MODEL"] = model
    env_data.pop("EDUX_API_OPT_OUT", None)

    with open(ENV_PATH, "w", encoding="utf-8") as f:
        for k, v in env_data.items():
            f.write(f"{k}={v}\n")


def ensure_login_gui() -> tuple[str, str, str, str]:
    load_env_file()
    email = os.environ.get("EDUX_EMAIL", "").strip()
    password = os.environ.get("EDUX_PASSWORD", "").strip()
    api_key = os.environ.get("EDUX_API_KEY", "").strip()
    model = os.environ.get("EDUX_MODEL", "").strip()

    if email and password and not is_legacy_env_file():
        return email, password, api_key, model

    # GUI for login
    root = tk.Tk()
    root.title("EDUX Login")
    root.geometry("800x600")
    root.resizable(True, True)
    root.attributes("-topmost", True)

    ui_font = ("Segoe UI", 10)
    title_font = ("Segoe UI", 11, "bold")

    tk.Label(root, text="Đăng nhập EDUX", font=title_font, fg="#2E7D32").pack(pady=(20, 10))

    msg = "Thông tin sẽ được lưu vào file .env trên máy của bạn để tránh việc phải nhập lại mỗi lần chạy."
    tk.Label(root, text=msg, font=("Segoe UI", 8), fg="gray", wraplength=520, justify="center").pack()

    content = tk.Frame(root)
    content.pack(fill="both", expand=True, padx=20, pady=10)

    tk.Label(content, text="Email:", font=ui_font).pack(pady=(15, 0))
    email_entry = tk.Entry(content, width=40, font=ui_font)
    email_entry.pack(pady=5)
    email_entry.insert(0, email)

    tk.Label(content, text="Mật khẩu:", font=ui_font).pack(pady=5)
    password_entry = tk.Entry(content, width=40, show="*", font=ui_font)
    password_entry.pack(pady=5)
    password_entry.insert(0, password)

    use_api_var = tk.BooleanVar(value=bool(api_key or model))
    use_api_checkbox = tk.Checkbutton(
        content,
        text="Sử dụng API key tự động cho test solver",
        font=ui_font,
        variable=use_api_var,
        onvalue=True,
        offvalue=False,
    )
    use_api_checkbox.pack(pady=(10, 0))

    api_frame = tk.Frame(content)

    tk.Label(api_frame, text="API key (tùy chọn):", font=ui_font).pack(pady=(10, 0))
    api_key_entry = tk.Entry(api_frame, width=40, show="*", font=ui_font)
    api_key_entry.pack(pady=5)
    api_key_entry.insert(0, api_key)

    tk.Label(api_frame, text="Model (tùy chọn):", font=ui_font).pack(pady=(10, 0))
    model_entry = tk.Entry(api_frame, width=40, font=ui_font)
    model_entry.pack(pady=5)
    model_entry.insert(0, model)

    def adjust_window_size() -> None:
        root.update_idletasks()
        req_w = root.winfo_reqwidth() + 40
        req_h = root.winfo_reqheight() + 40
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        width = min(max(req_w, 640), screen_w - 80)
        height = min(max(req_h, 420), screen_h - 80)
        x = (screen_w // 2) - (width // 2)
        y = (screen_h // 2) - (height // 2)
        root.geometry(f"{width}x{height}+{x}+{y}")

    def toggle_api_fields() -> None:
        if use_api_var.get():
            api_frame.pack()
        else:
            api_key_entry.delete(0, "end")
            model_entry.delete(0, "end")
            api_frame.pack_forget()
        adjust_window_size()

    use_api_checkbox.configure(command=toggle_api_fields)
    toggle_api_fields()

    result = {
        "email": "",
        "password": "",
        "api_key": "",
        "model": "",
        "submitted": False,
        "save": False,
    }

    def on_submit():
        e = email_entry.get().strip()
        p = password_entry.get().strip()
        if not e or not p:
            messagebox.showwarning("Thiếu thông tin", "Vui lòng nhập đầy đủ email và mật khẩu.")
            return
        result["email"] = e
        result["password"] = p
        if use_api_var.get():
            result["api_key"] = api_key_entry.get().strip()
            result["model"] = model_entry.get().strip()
        else:
            result["api_key"] = ""
            result["model"] = ""
        result["submitted"] = True
        result["save"] = True
        root.destroy()

    def on_self_login():
        result["email"] = ""
        result["password"] = ""
        if use_api_var.get():
            result["api_key"] = api_key_entry.get().strip()
            result["model"] = model_entry.get().strip()
        else:
            result["api_key"] = ""
            result["model"] = ""
        result["submitted"] = True
        result["save"] = False
        root.destroy()

    btn_frame = tk.Frame(content)
    btn_frame.pack(pady=10)

    tk.Button(
        btn_frame,
        text="Lưu & Đăng nhập",
        command=on_submit,
        bg="#4CAF50",
        fg="white",
        width=20,
        font=("Segoe UI", 10, "bold"),
        cursor="hand2",
    ).pack(pady=5)

    tk.Button(
        btn_frame,
        text="Tự đăng nhập (Không lưu)",
        command=on_self_login,
        bg="#757575",
        fg="white",
        width=20,
        font=("Segoe UI", 10),
        cursor="hand2",
    ).pack(pady=5)

    root.mainloop()

    if not result["submitted"]:
        print("[INFO] Đăng nhập bị hủy bởi người dùng.")
        sys.exit(0)

    if result["save"]:
        save_env_file(result["email"], result["password"], result["api_key"], result["model"])

    os.environ["EDUX_EMAIL"] = result["email"]
    os.environ["EDUX_PASSWORD"] = result["password"]
    os.environ["EDUX_API_KEY"] = result["api_key"]
    os.environ["EDUX_MODEL"] = result["model"]

    return result["email"], result["password"], result["api_key"], result["model"]
