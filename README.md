# Quizz Slayers - Bộ công cụ tự động hóa EDUX

Bộ công cụ giúp giải quyết các bài kiểm tra và slide bài giảng trên nền tảng EDUX một cách tự động, kết hợp giữa AI, OCR và Browser Automation.

## 🚀 Các công cụ chính

### 1. AI + OCR Slide Solver (`EDUX-SLIDE-AI`)
Sử dụng nhận diện hình ảnh (OCR) và trí tuệ nhân tạo (Ollama) để giải câu hỏi trực tiếp từ màn hình.
- **Cơ chế**: Chụp ảnh màn hình -> OCR nhận diện câu hỏi & đáp án -> AI (Arcee-VyLinh) chọn đáp án đúng -> Tự động click.
- **Ưu điểm**: Hoạt động không cần can thiệp vào trình duyệt, phù hợp cho các slide dạng canvas/hình ảnh khó inspect.
- **Yêu cầu**: Cần cài đặt [Ollama](https://ollama.com/) và tải model `hf.co/arcee-ai/Arcee-VyLinh-GGUF:Q8_0`.

### 2. Brute-force Slide Solver (`EDUX-SLIDE-BRUTEFORCE`)
Sử dụng Playwright để tự động tương tác với các slide bài giảng theo cơ chế thử sai (brute-force) hoặc ghi nhớ.
- **Cơ chế**: Tự động chuyển slide, tìm câu hỏi và thử các đáp án.
- **Ưu điểm**: Tốc độ cực nhanh, hoạt động ổn định trực tiếp trên trình duyệt.

### 3. Test Solver (`EDUX-TEST-SOLVER`)
Công cụ chuyên dụng để giải các bài kiểm tra (Tests) tập trung.
- **Cơ chế**: Tương tác qua Playwright, hỗ trợ nạp đáp án từ file `answers.txt` hoặc sử dụng prompt AI.
- **Định dạng `answers.txt`**: Mỗi dòng một đáp án theo định dạng `Số câu. Đáp án` (Ví dụ: `1. A`, `2. C`).
- **Ưu điểm**: Xử lý được các bài test dài, nhiều câu hỏi một cách tự động.

---

## 🛠 Cài đặt

### Yêu cầu hệ thống
- Python 3.10+
- Trình duyệt Chromium (sẽ được cài tự động qua Playwright)
- [Ollama](https://ollama.com/) (Dành cho AI Solver)

### Các bước cài đặt
1. Cài đặt các thư viện cơ bản cho Playwright tools:
   ```bash
   install_deps.bat
   ```
2. Cài đặt thư viện cho AI Solver:
   ```bash
   cd EDUX-SLIDE-AI
   pip install -r requirements.txt
   ```
3. Tải model cho Ollama (nếu sử dụng AI Solver):
   ```bash
   ollama run hf.co/arcee-ai/Arcee-VyLinh-GGUF:Q8_0
   ```

---

## 📖 Hướng dẫn sử dụng

### 1. Sử dụng AI Solver (Cho Slide)
1. Mở slide EDUX trên trình duyệt (để cửa sổ hiển thị trên màn hình).
2. Chạy lệnh:
   ```bash
   python EDUX-SLIDE-AI/scripts/main.py
   ```
   *Nhấn phím `S` để bắt đầu/tạm dừng.*

### 2. Sử dụng Brute-force Solver (Cho Slide)
Chạy trực tiếp file batch:
```bash
run_slide_bruteforce.bat
```

### 3. Sử dụng Test Solver (Cho Bài Kiểm Tra)
1. Chuẩn bị file đáp án (nếu có) hoặc cấu hình prompt.
2. Chạy file batch:
```bash
run_test_solver.bat
```

---

## ⚙️ Cấu hình
Thông tin đăng nhập được lưu tại file `.env` ở thư mục gốc của mỗi tool (hoặc sẽ yêu cầu nhập ở lần đầu chạy).

```env
EDUX_EMAIL=your_email@example.com
EDUX_PASSWORD=your_password
```

---
*Lưu ý: Công cụ này được tạo ra cho mục đích nghiên cứu và học tập. Vui lòng sử dụng có trách nhiệm.*
