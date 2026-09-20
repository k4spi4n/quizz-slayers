# ⚡ Quizz Slayers - Bộ công cụ tự động hóa EDUX

> 📢 **THÔNG BÁO QUAN TRỌNG**: 
> - **Tiện ích mở rộng trình duyệt (`EDUX-EXTENSION`) hiện là trọng tâm phát triển chính (Main Development Focus)** của dự án. Mọi tính năng mới, tối ưu hóa và cập nhật trong tương lai sẽ tập trung hoàn toàn vào Extension này.
> - **Các script chạy bằng Playwright / Python (`EDUX-SLIDE-BRUTEFORCE`, `EDUX-TEST-SOLVER`, `EDUX-LIVE-QUESTION`, `EDUX-SLIDE-AI`) hiện đã bị NGỪNG HỖ TRỢ (DEPRECATED)**. Người dùng được khuyến nghị chuyển đổi 100% sang sử dụng Extension để có trải nghiệm mượt mà, tiện lợi, không cần cài đặt môi trường phức tạp và không lo vấn đề session/đăng nhập.

---

## 🌟 Trọng tâm chính: EDUX Slayers Browser Extension (`EDUX-EXTENSION`)

Tiện ích mở rộng trình duyệt (Manifest V3) dành cho Google Chrome, Microsoft Edge, Brave, Cốc Cốc, Opera,... Tích hợp toàn diện các tính năng giải Slide, giải Bài tập (Test), theo dõi điểm số và tiến trình học tập trực tiếp trên nền tảng EDUX.

### ✨ Ưu điểm vượt trội so với script Playwright cũ
- ⚡ **Không cần cài Python / Playwright**: Cài đặt 1 lần trực tiếp vào trình duyệt trong 30 giây.
- 🔐 **Không lo phiên đăng nhập**: Sử dụng trực tiếp phiên đăng nhập (session/cookies) hiện tại của bạn trên trình duyệt, không cần lưu tài khoản/mật khẩu vào file `.env`.
- 🧠 **Tích hợp AI đa nền tảng**: Hỗ trợ Google Gemini, OpenAI, DeepSeek, OpenRouter, Ollama (Local) và Custom Endpoint.
- 🎯 **Giao diện trực quan**: Popup điều khiển tiện lợi, widget nổi ngay trên trang bài giảng và thanh tiến độ trực quan trên trang môn học.
- 🛡️ **Ổn định & An toàn**: Tự động nhận diện cấu trúc DOM mới nhất của EDUX, cơ chế AI phân tích trước khi click kèm Fallback thử sai (brute-force) thông minh.

---

## 📦 Tải về (Releases)

Bạn có thể tải tiện ích mở rộng theo 2 cách:

### Cách 1: Tải file nén đóng gói sẵn từ GitHub Releases (Khuyên dùng)
1. Truy cập mục **[Releases](../../releases)** của repository.
2. Tải về file nén `edux-extension.zip` mới nhất từ mục **Assets**.
3. Giải nén file `edux-extension.zip` vào một thư mục trên máy tính của bạn (ví dụ: `C:\edux-extension`).

### Cách 2: Sử dụng trực tiếp từ mã nguồn repository
Nếu bạn đã clone hoặc tải mã nguồn repository này về máy tính:
- Thư mục extension nằm sẵn tại: `EDUX-EXTENSION/`.

---

## 🛠 Hướng dẫn Cài đặt Extension vào Trình duyệt

Hỗ trợ tất cả các trình duyệt nhân Chromium: **Google Chrome, Microsoft Edge, Brave, Cốc Cốc, Opera,...**

1. **Mở trang quản lý Tiện ích mở rộng (Extensions)** trên trình duyệt:
   - **Google Chrome**: Nhập `chrome://extensions/` vào thanh địa chỉ.
   - **Microsoft Edge**: Nhập `edge://extensions/` vào thanh địa chỉ.
   - **Brave**: Nhập `brave://extensions/` vào thanh địa chỉ.
   - **Cốc Cốc**: Nhập `coccoc://extensions/` vào thanh địa chỉ.
2. **Bật Chế độ dành cho nhà phát triển (Developer mode)**:
   - Gạt công tắc **Developer mode** (thường nằm ở góc trên bên phải trang quản lý tiện ích).
3. **Cài đặt tiện ích**:
   - Nhấn vào nút **Tải tiện ích đã giải nén** (*Load unpacked*).
   - Chọn thư mục đã giải nén ở bước trước (hoặc chọn thư mục `EDUX-EXTENSION` trong repo).
4. **Ghim tiện ích**:
   - Nhấn vào biểu tượng mảnh ghép (Extensions) trên thanh công cụ trình duyệt và **Ghim (Pin)** biểu tượng **EDUX Slayers** ⚔️ để thuận tiện sử dụng.

---

## 📖 Hướng dẫn Sử dụng Extension

### 1. Cấu hình AI (Tab ⚙️ Cài đặt)
Để kích hoạt tính năng giải tự động bằng AI, bạn chỉ cần thiết lập API 1 lần:
1. Click vào biểu tượng **EDUX Slayers** ⚔️ trên thanh công cụ trình duyệt.
2. Chuyển sang tab **⚙️ Cài đặt**.
3. Chọn nhà cung cấp (**Preset**):
   - **Google Gemini**: Nhập Gemini API Key (Model mặc định: `gemini-2.0-flash`).
   - **OpenAI**: Nhập OpenAI API Key (Model: `gpt-4o-mini`, `gpt-4o`,...).
   - **DeepSeek**: Nhập DeepSeek API Key (`deepseek-chat`).
   - **OpenRouter**: Nhập OpenRouter API Key.
   - **Ollama (Local)**: Chạy Ollama trên máy tính (`http://localhost:11434/v1`).
   - **Tùy chỉnh (Custom)**: Nhập Base URL riêng (mặc định hỗ trợ các proxy / server local chuẩn OpenAI API như `http://localhost:20128/v1`). Bấm nút **🔄 Lấy DS** để tự động lấy danh sách models.
4. Bấm **💾 Lưu cài đặt**.

### 2. Tự động giải Slide bài giảng
1. Đăng nhập vào trang web EDUX và mở slide bài giảng bạn cần học.
2. Click icon **EDUX Slayers** ⚔️.
3. Nhấn **▶️ Bắt đầu giải Slide**.
   - Extension sẽ tự động đọc câu hỏi và các lựa chọn trên slide.
   - Gửi đề tới AI, **chờ AI phân tích trả về đáp án đúng** rồi mới click chọn.
   - Nếu AI trả lời sai hoặc gặp sự cố, hệ thống sẽ tự động chuyển sang cơ chế thử sai (Fallback brute-force) để đảm bảo không bị dừng bài.
   - Tự động nhận diện các nút `Kiểm tra`, `Thử lại`, `Câu tiếp theo`, `Trang sau` để chuyển tiếp cho đến khi hoàn thành bài giảng.

### 3. Tự động giải Bài tập (Test Solver)
1. Mở bài tập EDUX cần làm (hoặc bấm nút `🚀 Mở bài` trong tab Bài tập của Extension).
2. Mở Extension ➔ Chuyển sang Tab **📝 Bài tập**.
3. Chọn một trong 2 phương thức giải:
   - **Cách 1: Giải tự động 100% bằng AI (Khuyên dùng)**:
     - Bấm nút **⚡ Giải AI**. Extension sẽ tự động bắt đề thi, gửi tới mô hình AI đã cấu hình, nhận diện câu trả lời (trắc nghiệm, đúng/sai, điền khuyết, tự luận), tự động điền câu trả lời và nộp bài.
   - **Cách 2: Giải thủ công qua Chatbot AI (ChatGPT, Claude, Gemini Web)**:
     - Bấm nút **📋 Copy Prompt** để sao chép toàn bộ đề bài kèm hướng dẫn chuẩn vào Clipboard.
     - Dán prompt vào trang chat AI của bạn (ChatGPT, Claude, v.v.) và sao chép kết quả AI trả lời.
     - Trở lại popup Extension, bấm **📥 Dán Clipboard** (hoặc dán vào ô nhập liệu).
     - Bấm **✨ Bắt đầu điền bài tập**.

### 4. Theo dõi Tiến trình & Điểm số bài tập
- **Trang chủ học phần (`/student`)**: Hiển thị thanh tiến trình trực quan (% Slide hoàn thành, % Bài tập AI đã làm, cảnh báo số bài tập còn thiếu) ngay trên từng thẻ môn học.
- **Trang chi tiết môn học (`/subject?id=...`)**: Hiển thị điểm số cao nhất (`🏆 Điểm: X/10`) và cảnh báo bài chưa làm (`⚠️ Chưa làm`) bên cạnh các bài tập.
- **Tab 📊 Điểm số trong Extension**: Theo dõi bảng điểm tổng hợp của tất cả các bài tập trong môn học đang mở.

---

## 📦 Các công cụ cũ / Playwright Scripts [DEPRECATED - NGỪNG HỖ TRỢ]

> ⚠️ **Lưu ý**: Các công cụ Python/Playwright dưới đây **không còn được bảo trì hoặc cập nhật**. Vui lòng sử dụng **EDUX-EXTENSION** ở trên để có tính năng mới nhất và độ ổn định cao nhất.

<details>
<summary><b>Xem hướng dẫn các script cũ (Click để mở)</b></summary>

### 1. Brute-force Slide Solver (`EDUX-SLIDE-BRUTEFORCE`)
Sử dụng Playwright để giải slide theo cơ chế thử sai:
```bash
run_slide_bruteforce.bat
```

### 2. Test Solver (`EDUX-TEST-SOLVER`)
Sử dụng Playwright để tự động điền bài test từ file `answers.txt`:
```bash
run_test_solver.bat
```

### 3. AI + OCR Slide Solver (`EDUX-SLIDE-AI`)
Sử dụng OCR chụp màn hình và mô hình Ollama cục bộ:
```bash
cd EDUX-SLIDE-AI
pip install -r requirements.txt
ollama run hf.co/arcee-ai/Arcee-VyLinh-GGUF:Q8_0
python scripts/main.py
```

### 4. Live Solver (`EDUX-LIVE-QUESTION`)
Giải câu hỏi trực tiếp trên lớp học:
```bash
run_live_solver.bat
```

### Yêu cầu hệ thống cho script cũ:
- Python 3.10+
- Chạy `install_deps.bat` để cài đặt dependencies.
- File `.env` chứa thông tin đăng nhập:
  ```env
  EDUX_EMAIL=your_email@example.com
  EDUX_PASSWORD=your_password
  ```
</details>

---

## 📜 Điều khoản & Miễn trừ trách nhiệm
*Công cụ này được phát triển phục vụ mục đích nghiên cứu và học tập. Người dùng tự chịu trách nhiệm về mục đích và hành vi khi sử dụng công cụ trên nền tảng EDUX.*
