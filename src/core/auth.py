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

def save_env_file(email: str, password: str) -> None:
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
    
    with open(ENV_PATH, "w", encoding="utf-8") as f:
        for k, v in env_data.items():
            f.write(f"{k}={v}\n")

def ensure_login_gui() -> tuple[str, str]:
    load_env_file()
    email = os.environ.get("EDUX_EMAIL", "").strip()
    password = os.environ.get("EDUX_PASSWORD", "").strip()

    if email and password:
        return email, password

    # GUI for login
    root = tk.Tk()
    root.title("EDUX Login")
    root.geometry("800x400")
    root.resizable(False, False)
    root.attributes("-topmost", True)

    ui_font = ("Segoe UI", 10)
    title_font = ("Segoe UI", 11, "bold")
    
    tk.Label(root, text="Đăng nhập EDUX", font=title_font, fg="#2E7D32").pack(pady=(20, 10))
    
    msg = "Thông tin sẽ được lưu vào file .env trên máy của bạn để tránh việc phải nhập lại mỗi lần chạy."
    tk.Label(root, text=msg, font=("Segoe UI", 8), fg="gray").pack()

    tk.Label(root, text="Email:", font=ui_font).pack(pady=(15, 0))
    email_entry = tk.Entry(root, width=40, font=ui_font)
    email_entry.pack(pady=5)
    email_entry.insert(0, email)

    tk.Label(root, text="Mật khẩu:", font=ui_font).pack(pady=5)
    password_entry = tk.Entry(root, width=40, show="*", font=ui_font)
    password_entry.pack(pady=5)
    password_entry.insert(0, password)

    result = {"email": "", "password": "", "submitted": False, "save": False}

    def on_submit():
        e = email_entry.get().strip()
        p = password_entry.get().strip()
        if not e or not p:
            messagebox.showwarning("Thiếu thông tin", "Vui lòng nhập đầy đủ email và mật khẩu.")
            return
        result["email"] = e
        result["password"] = p
        result["submitted"] = True
        result["save"] = True
        root.destroy()

    def on_self_login():
        result["email"] = ""
        result["password"] = ""
        result["submitted"] = True
        result["save"] = False
        root.destroy()

    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=10)

    tk.Button(btn_frame, text="Lưu & Đăng nhập", command=on_submit, bg="#4CAF50", fg="white", 
              width=20, font=("Segoe UI", 10, "bold"), cursor="hand2").pack(pady=5)
    
    tk.Button(btn_frame, text="Tự đăng nhập (Không lưu)", command=on_self_login, bg="#757575", fg="white", 
              width=20, font=("Segoe UI", 10), cursor="hand2").pack(pady=5)

    # Center window
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')

    root.mainloop()

    if not result["submitted"]:
        print("[INFO] Đăng nhập bị hủy bởi người dùng.")
        sys.exit(0)

    if result["save"]:
        save_env_file(result["email"], result["password"])
    
    os.environ["EDUX_EMAIL"] = result["email"]
    os.environ["EDUX_PASSWORD"] = result["password"]
    
    return result["email"], result["password"]
