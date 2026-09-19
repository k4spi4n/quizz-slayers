# ⚔️ EDUX Slayers Browser Extension (Plugin Trình Duyệt)

Bộ công cụ tự động hóa giải Slide bài giảng & Bài tập trên nền tảng EDUX dưới dạng **Chrome/Edge Extension (Manifest V3)**.

---

## 🚀 Tính năng nổi bật

1. **🧠 AI + Fallback Slide Solver (Tự động giải Slide bằng AI)**:
   - **Tự động gửi câu hỏi tới AI**: Trích xuất đề câu hỏi và các lựa chọn trên slide, gửi đến AI để phân tích và chọn đáp án chính xác nhất.
   - **Đảm bảo an toàn - Chờ AI trả về mới chọn**: Đợi AI phân tích xong mới click để tránh chọn nhầm hoặc vội vàng gây lỗi.
   - **Cơ chế Fallback thông minh**: Nếu AI trả lời chưa đúng hoặc gặp lỗi kết nối, hệ thống tự động chuyển sang cơ chế thử sai (brute-force) để không bao giờ bị dừng tiến trình.
   - Tự động nhận diện các nút: `Trả lời trên lớp`, `Kiểm tra`, `Thử lại`, `Câu tiếp theo`, `Trang sau`.
   - Tự động chuyển trang khi hoàn thành slide hoặc slide không có câu hỏi.

2. **📝 Test Solver (Tự động giải Bài Tập - Mô phỏng EDUX-TEST-SOLVER)**:
   - **Tự động bắt đề bài tập**: Lắng nghe phản hồi từ máy chủ khi nhấn nút `Làm bài tập` trên EDUX.
   - **Chuẩn hóa Prompt câu hỏi**: Cấu trúc JSON gọn gàng kèm hướng dẫn chuẩn format của EDUX-TEST-SOLVER.
   - **Giải bài bằng AI tích hợp linh hoạt**: Hỗ trợ gọi trực tiếp API Gemini (`gemini-2.0-flash`), OpenAI (`gpt-4o-mini`), DeepSeek (`deepseek-chat`), OpenRouter, Ollama local (`localhost:11434`), hoặc tùy chỉnh Endpoint/Base URL riêng chỉ với 1 click (`⚡ Giải AI`).
   - **Hỗ trợ giải thủ công linh hoạt**:
     - `📋 Copy Prompt`: Sao chép prompt vào Clipboard để dán vào bất kỳ Chatbot AI nào (ChatGPT, Claude, Gemini).
     - `📥 Dán Clipboard`: Tự động nạp kết quả trả về từ AI vào ô đáp án.
   - **Parse đáp án đa định dạng**: Nhận diện JSONL 1 dòng, JSON Array, JSON Object, concatenated JSON `}{`, hoặc dạng văn bản `1. A, 2. B`.
   - **Điền bài từng bước chuẩn xác**: Tự động nhận diện Đúng/Sai, tự luận, điền ô trống, trắc nghiệm A/B/C/D, chờ chuyển câu và tự động click `Nộp bài`.

3. **📊 Bảng Điểm & Cảnh Báo Bài Tập AI**:
   - Tự động lấy điểm số cao nhất của các bài tập (`[Bài tập AI]`) và hiển thị huy hiệu trực tiếp (`🏆 Điểm: X/10`).
   - Cảnh báo trực quan (`⚠️ Chưa làm`) cho các bài tập chưa nộp để tránh bỏ sót.
   - Banner tổng quan tiến độ môn học hiển thị ngay đầu danh sách bài học kèm nút cuộn nhanh tới bài chưa làm.
   - Tab **📊 Điểm số** trong Popup tiện ích giúp theo dõi toàn diện tiến độ của môn học hiện tại.

4. **🎯 Theo Dõi Tiến Trình Học Phần Trực Quan Từ Bên Ngoài (/student)**:
   - Hiển thị trực tiếp thanh tiến trình sinh động ngay trên từng thẻ học phần ở trang chủ sinh viên:
     - 🖥️ **Slide**: Số slide đã đọc / tổng số slide (`X/Y (Z%)`) kèm thanh phần trăm màu xanh.
     - 📝 **Bài tập**: Số bài tập AI đã làm / tổng số bài tập (`A/B (C%)`) kèm thanh phần trăm màu tím/hồng.
     - 🏷️ **Huy hiệu thông minh**: `✓ Đã xong 100%` (xanh lá), `⚠️ Còn X bài tập` (đỏ/hồng), `📖 Còn Y slide` (vàng).
     - Rê chuột vào thẻ để xem tooltip chi tiết: số bài giảng đã xong, điểm số trung bình, điểm cao nhất.
   - Hỗ trợ xem tổng quan toàn bộ các học phần trong tab **📊 Điểm số** của Popup.

---

## 🛠 Hướng dẫn Cài đặt vào Trình duyệt (Chrome / Edge / Brave / Opera)

1. Mở trình duyệt web của bạn.
2. Truy cập trang quản lý extension:
   - **Google Chrome**: Gõ `chrome://extensions/` lên thanh địa chỉ.
   - **Microsoft Edge**: Gõ `edge://extensions/` lên thanh địa chỉ.
   - **Brave**: Gõ `brave://extensions/` lên thanh địa chỉ.
3. Bật chế độ nhà phát triển (**Developer mode**) ở góc trên bên phải màn hình.
4. Nhấn nút **Tải tiện ích đã giải nén** (*Load unpacked*).
5. Trỏ tới thư mục: `D:\CODE\quizz-slayers\EDUX-EXTENSION` và nhấn **Select Folder**.

---

## 📖 Hướng dẫn Sử dụng

### 1. Giải Slide bài giảng tự động (Bằng AI):
- Đăng nhập vào trang web EDUX trên trình duyệt của bạn như bình thường.
- Mở slide bài giảng đang học.
- Đảm bảo đã cấu hình API ở tab **⚙️ Cài đặt** để bật chế độ AI siêu chuẩn xác.
- Click icon **EDUX Slayers** ⚔️ ở góc trình duyệt.
- Nhấn **▶️ Bắt đầu giải Slide**. Extension sẽ tự động trích xuất câu hỏi, gửi AI phân tích, CHỜ AI trả về đáp án chuẩn xác rồi mới click (kèm fallback thử sai nếu cần).

### 2. Giải bài tập (Test Solver):
- Mở trang bài tập EDUX (hoặc bấm nút `🚀 Mở bài` trên extension).
- Mở Extension ➔ Chuyển sang Tab **📝 Bài tập**.
- **Cách 1: Giải tự động hoàn toàn bằng AI (Khuyên dùng)**:
  - Vào Tab **⚙️ Cài đặt**:
    - Chọn cấu hình mẫu (**Preset**): Google Gemini, OpenAI, DeepSeek, OpenRouter, Ollama (Local), hoặc **Tùy chỉnh (Custom)**.
    - Khi chọn **Tùy chỉnh**, khung nhập URL nguồn API sẽ hiện ra với mặc định: `http://localhost:20128/v1`.
    - Bấm nút **🔄 Lấy DS** để tự động kéo danh sách models từ server (chuẩn OpenAI-compatible `GET /models`).
    - Nhập `API Key` tương ứng (nếu dùng server local thì để trống).
    - Bấm **💾 Lưu cài đặt**.
  - Trở lại Tab **📝 Bài tập** ➔ Bấm **⚡ Giải AI**. Extension sẽ tự lấy đề, gửi AI giải, điền đáp án và nộp bài.
- **Cách 2: Giải thủ công qua Chatbot AI (ChatGPT/Claude/Gemini web)**:
  - Bấm **📋 Copy Prompt** để sao chép toàn bộ câu hỏi và lệnh chuẩn vào Clipboard.
  - Dán vào Chatbot AI trên trình duyệt và gửi.
  - Sao chép câu trả lời của AI ➔ Trở lại Extension bấm **📥 Dán Clipboard** (hoặc dán tay vào ô).
  - Bấm **✨ Bắt đầu điền bài tập**.

### 3. Theo dõi điểm số bài tập:
- Mở trang môn học EDUX (`/subject?id=...`).
- Xem điểm số cao nhất và cảnh báo chưa làm ngay bên cạnh các nút `[Bài tập AI]`.
- Hoặc mở Popup extension ➔ Chuyển sang Tab **📊 Điểm số** để xem thống kê chi tiết.

---

*Lưu ý: Công cụ này được tạo ra cho mục đích nghiên cứu và học tập. Vui lòng sử dụng có trách nhiệm.*
