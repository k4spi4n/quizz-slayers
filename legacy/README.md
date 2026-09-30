# 📦 Script Python/Playwright cũ (ngừng hỗ trợ)

> [!WARNING]
> Các script trong thư mục này **không còn được bảo trì** và có thể không chạy được với giao diện EDUX hiện tại.
> Hãy dùng **[Extension](../EDUX-EXTENSION/)** — cài trong 30 giây, không cần Python, không cần lưu mật khẩu.

| Thư mục | Chức năng cũ |
| --- | --- |
| `EDUX-SLIDE-BRUTEFORCE/` | Giải slide bằng cách thử lần lượt từng đáp án (Playwright) |
| `EDUX-TEST-SOLVER/` | Điền bài tập từ `answers.txt` (Playwright) |
| `EDUX-SLIDE-AI/` | Giải slide bằng OCR + Ollama |
| `EDUX-LIVE-QUESTION/` | Giải câu hỏi trực tiếp |
| `src/core/auth.py` | Đăng nhập dùng chung, lưu thông tin vào `.env` |

## Nếu vẫn muốn chạy

Yêu cầu: Python 3.10+.

```bat
cd legacy
install_deps.bat           :: cài thư viện + Playwright Chromium
run_slide_bruteforce.bat   :: EDUX-SLIDE-BRUTEFORCE
run_test_solver.bat        :: EDUX-TEST-SOLVER (điền từ answers.txt)
run_live_solver.bat        :: EDUX-LIVE-QUESTION
:: EDUX-SLIDE-AI: cd EDUX-SLIDE-AI && pip install -r requirements.txt
```

Thông tin đăng nhập (`EDUX_EMAIL` / `EDUX_PASSWORD`) được lưu trong **`legacy/.env`**. Nếu trước đây bạn có file `.env` ở thư mục gốc repo, hãy chuyển nó vào `legacy/`.
