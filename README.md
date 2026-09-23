# ⚡ Quizz Slayers — Tự động hóa EDUX

> Extension Chromium (Manifest V3) giải **Slide**, **Bài tập** và theo dõi **Điểm số** ngay trên EDUX — không cần Python, không cần lưu mật khẩu.

![version](https://img.shields.io/badge/version-v2.3.1-blue) ![mv3](https://img.shields.io/badge/manifest-V3-green) ![chromium](https://img.shields.io/badge/Chrome%20%7C%20Edge%20%7C%20Brave%20%7C%20C%E1%BB%91c%20C%E1%BB%91c-orange)

> [!IMPORTANT]
> **Extension (`EDUX-EXTENSION`) là trọng tâm phát triển duy nhất.** Các script Python/Playwright (`EDUX-SLIDE-BRUTEFORCE`, `EDUX-TEST-SOLVER`, `EDUX-SLIDE-AI`, `EDUX-LIVE-QUESTION`) đã **ngừng hỗ trợ** — xem phần cuối README.

---

## ✨ Tính năng

| Tab | Làm được gì |
| --- | --- |
| ⚡ Slide | Tự động đọc câu hỏi → AI phân tích → click đáp án → tự chuyển trang. 2 chế độ: 🧠 **AI Chuẩn xác** (cần API key) và ⚡ **Thử sai nhanh** (không cần key). |
| 📝 Bài tập | 2 cách giải: ⚡ **Tự động (API)** — 1 chạm bắt đề → AI giải → điền → nộp; 💬 **Chatbot** — copy đề sang ChatGPT/Gemini/Claude web (miễn phí, không cần key) rồi dán đáp án về. |
| 📊 Điểm số | Quét tiến độ Slide + Bài tập, điểm cao nhất từng bài, cảnh báo bài chưa làm — cả ở trang môn học lẫn trong popup. |
| ⚙️ Cài đặt | Cấu hình 1 lần: provider, model, API key, delay, tự nộp bài, tự chuyển slide. |

**Vì sao dùng Extension thay script cũ:** cài trong 30 giây · dùng luôn session đăng nhập trên trình duyệt · hỗ trợ Gemini / OpenAI / DeepSeek / OpenRouter / Ollama / Custom endpoint · popup + widget trực quan · tự thích ứng DOM EDUX mới.

---

## 📦 Tải & cài đặt (1 phút)

**Cách 1 — Releases (khuyên dùng):** tải `edux-extension.zip` từ mục **[Releases](../../releases)** → giải nén (VD: `C:\edux-extension`).

**Cách 2 — Từ mã nguồn:** dùng sẵn thư mục `EDUX-EXTENSION/` trong repo này.

Sau đó trên Chrome / Edge / Brave / Cốc Cốc / Opera:

1. Mở `chrome://extensions` (Edge: `edge://extensions`, Brave: `brave://extensions`, Cốc Cốc: `coccoc://extensions`) → bật **Developer mode**.
2. **Load unpacked** → chọn thư mục đã giải nén (hoặc `EDUX-EXTENSION/`).
3. Ghim **EDUX Slayers** ⚔️ ra thanh công cụ.

---

## 📖 Sử dụng

### 1. Cấu hình AI một lần (tab ⚙️ Cài đặt)

Mở popup → tab **Cài đặt** → chọn provider → nhập key/model → **💾 Lưu cài đặt**.

| Provider | Key / Endpoint |
| --- | --- |
| Google Gemini (mặc định) | `AIza...` · `gemini-2.0-flash` |
| OpenAI | `sk-...` · `gpt-4o-mini` / `gpt-4o` |
| DeepSeek | `sk-...` · `deepseek-chat` |
| OpenRouter | `sk-or-v1-...` |
| Ollama (local, offline) | `http://localhost:11434/v1` · không cần key |
| Custom | Base URL chuẩn OpenAI (VD: `http://localhost:20128/v1`) · nút **🔄 Lấy DS** để load models |

### 2. Giải Slide (tab ⚡ Slide)

1. Mở slide bài giảng EDUX.
2. Mở popup → chọn **🧠 AI Chuẩn xác** hoặc **⚡ Thử sai nhanh**.
3. Bấm **▶️ Bắt đầu giải Slide** — Extension tự trả lời, bấm `Kiểm tra` / `Câu tiếp theo` / `Trang sau`, hết bài thì **⏹️ Dừng lại**.

### 3. Giải Bài tập (tab 📝 Bài tập)

Mở bài tập EDUX (hoặc bấm **🚀 Mở bài** trong popup) → tab **Bài tập**:

- **⚡ Tự động (API):** bấm **GIẢI BÀI TẬP BẰNG AI** → theo dõi 3 bước `Bắt đề → AI giải → Điền & Nộp`. Muốn đổi model thì bấm **⚙️ Đổi Model**.
- **💬 Chatbot (không cần key):**
  1. **Bước 1:** bấm **📋 Copy đề & xem trước** → đề được copy sẵn, có thể mở nhanh ChatGPT / Gemini / Claude ngay trong popup.
  2. **Bước 2:** dán đề vào chatbot, copy đáp án → về popup bấm **📥 Dán Clipboard** (lần đầu trình duyệt sẽ hỏi quyền clipboard → chọn **Cho phép**) hoặc `Ctrl+V` vào ô.
  3. **Bước 3:** bấm **✨ Bắt đầu điền bài tập**.
- Nút **🔄 Phiên mới** để xóa đáp án cũ khi làm bài khác.

### 4. Xem điểm (tab 📊 Điểm số)

- Trang `/student`: thanh tiến độ ngay trên từng thẻ môn học.
- Trang môn học: huy hiệu `🏆 X/10` hoặc `⚠️ Chưa làm` cạnh mỗi bài.
- Popup → tab **Điểm số** → **🔄 Quét lại** để xem bảng tổng hợp.

---

## 🗂 Cấu trúc repo

```text
EDUX-EXTENSION/          # Extension chính (duy nhất còn phát triển)
├── manifest.json        # Manifest V3, v2.3.1
├── popup/               # Giao diện popup (Slide / Bài tập / Điểm số / Cài đặt)
├── scripts/             # slide-solver, test-solver, score-tracker, dom-utils
├── content.js / injected.js / background.js
EDUX-SLIDE-BRUTEFORCE/   # [deprecated] script Playwright cũ
EDUX-TEST-SOLVER/        # [deprecated] script Playwright cũ
EDUX-SLIDE-AI/           # [deprecated] script OCR + Ollama cũ
EDUX-LIVE-QUESTION/      # [deprecated] script cũ
edux-extension.zip       # Bản đóng gói sẵn
```

---

<details>
<summary><b>📦 Script Python/Playwright cũ (ngừng hỗ trợ — bấm để xem)</b></summary>

Không còn bảo trì. Hãy dùng Extension ở trên.

```bash
run_slide_bruteforce.bat   # EDUX-SLIDE-BRUTEFORCE
run_test_solver.bat        # EDUX-TEST-SOLVER (điền từ answers.txt)
run_live_solver.bat        # EDUX-LIVE-QUESTION
# EDUX-SLIDE-AI: cd EDUX-SLIDE-AI && pip install -r requirements.txt
```

Yêu cầu cũ (nếu vẫn cố dùng): Python 3.10+, `install_deps.bat`, file `.env` với `EDUX_EMAIL` / `EDUX_PASSWORD`.

</details>

---

## 📜 Miễn trừ trách nhiệm

_Công cụ phục vụ mục đích nghiên cứu và học tập. Người dùng tự chịu trách nhiệm khi sử dụng trên nền tảng EDUX._
