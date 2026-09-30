# ⚡ Quizz Slayers — Tự động hóa EDUX

> Extension Chromium (Manifest V3) giải **Slide**, **Bài tập** và theo dõi **Điểm số** ngay trên EDUX — không cần Python, không cần lưu mật khẩu.

![version](https://img.shields.io/badge/version-v2.6.0-blue) ![mv3](https://img.shields.io/badge/manifest-V3-green) ![chromium](https://img.shields.io/badge/Chrome%20%7C%20Edge%20%7C%20Brave%20%7C%20C%E1%BB%91c%20C%E1%BB%91c-orange)

> [!IMPORTANT]
> **Extension (`EDUX-EXTENSION`) là trọng tâm phát triển duy nhất.** Các script Python/Playwright (`EDUX-SLIDE-BRUTEFORCE`, `EDUX-TEST-SOLVER`, `EDUX-SLIDE-AI`, `EDUX-LIVE-QUESTION`) đã **ngừng hỗ trợ** và được chuyển vào [`legacy/`](legacy/).

---

## ✨ Tính năng

| Tab | Làm được gì |
| --- | --- |
| ⚡ Slide | Tự động đọc câu hỏi → AI phân tích → click đáp án → tự chuyển trang. 3 chế độ: 🧠 **AI** (cần API key), 🎯 **Laya** 🧪 *thử nghiệm* (model local, cân bằng tốc độ/độ chính xác) và ⚡ **Bruteforce** (không cần key). |
| 📝 Bài tập | 2 cách giải: ⚡ **Tự động (API)** — 1 chạm bắt đề → AI giải → điền → nộp; 💬 **Chatbot** — copy đề sang ChatGPT/Gemini/Claude web (miễn phí, không cần key) rồi dán đáp án về. |
| 📊 Điểm số | Quét tiến độ Slide + Bài tập, điểm cao nhất từng bài, cảnh báo bài chưa làm — cả ở trang môn học lẫn trong popup. |
| ⚙️ Cài đặt | Cấu hình 1 lần: provider, model, API key, delay, tự nộp bài, thời gian chờ mỗi câu bài tập, tự chuyển slide. Kiểm tra bản mới, sao lưu / khôi phục cấu hình. |

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

## 🔄 Cập nhật (giữ nguyên cấu hình AI & API key)

Cấu hình được trình duyệt lưu theo extension. **Chép đè file + Reload** thì giữ nguyên; **Remove** extension hoặc **Load unpacked từ thư mục khác** thì mất sạch.

Khi có bản mới, icon extension hiện nhãn **NEW** và popup báo 🎉 **Có bản mới**. Để cập nhật:

1. Mở thư mục đã cài extension → chạy **`update.bat`** (tự tải bản mới nhất và chép đè vào đúng thư mục đó).
   Hoặc thủ công: tải `edux-extension.zip` ở [Releases](../../releases/latest) → giải nén **đè lên thư mục cũ** (chọn *Replace*).
2. Mở popup → bấm **🔄 Áp dụng** (hoặc nút ↻ Reload ở trang `chrome://extensions`).

> [!TIP]
> Trước khi cài lại, đổi thư mục, đổi trình duyệt hay đổi máy: vào **⚙️ Cài đặt → 🔄 Cập nhật & Sao lưu → 📤 Xuất file** để lưu cấu hình ra file `.json`, cài xong bấm **📥 Khôi phục**. File chứa API key — đừng chia sẻ.

> [!NOTE]
> Từ **v2.4.0 trở về trước** chưa có `update.bat`: lần này hãy giải nén zip mới **đè lên thư mục cũ** rồi bấm ↻ Reload. Những lần sau chỉ cần chạy `update.bat`. Nếu cài từ mã nguồn git thì dùng `git pull` rồi Reload.

---

## 📖 Sử dụng

### 1. Cấu hình AI (tab ⚙️ Cài đặt)

Mở popup → tab **Cài đặt** → **🤖 Cấu hình AI** → **+ Thêm** → chọn provider, model, nhập key → **💾 Lưu cấu hình**.

- Thêm được **nhiều cấu hình** (nhiều provider / nhiều key), sửa ✏️ hoặc xóa 🗑 (bấm 2 lần).
- Mỗi chức năng chọn cấu hình riêng: **Slide (AI) dùng** và **Bài tập (API) dùng** — VD Slide dùng Mercury 2.5 cho nhanh, Bài tập dùng Gemini. Tab Slide (chế độ AI) và tab Bài tập cũng có ô 🤖 chọn nhanh.
- Inception (Mercury) có thêm **Mức suy luận**: `instant` / `low` / `medium` / `high` (mặc định `medium`). VD tạo 2 cấu hình: Mercury `instant` cho Slide, Mercury `high` cho Bài tập.
- Cấu hình cũ (1 provider) được tự chuyển thành cấu hình đầu tiên khi mở popup.

| Provider | Key / Endpoint |
| --- | --- |
| Google Gemini (mặc định) | `AIza...` · `gemini-2.0-flash` |
| OpenAI | `sk-...` · `gpt-4o-mini` / `gpt-4o` |
| DeepSeek | `sk-...` · `deepseek-chat` |
| OpenRouter | `sk-or-v1-...` |
| Inception | API key · `mercury-2.5` |
| Ollama (local, offline) | `http://localhost:11434/v1` · không cần key |
| Custom | Base URL chuẩn OpenAI (VD: `http://localhost:20128/v1`) · nút **🔄 Lấy DS** để load models |

### 2. Giải Slide (tab ⚡ Slide)

1. Mở slide bài giảng EDUX.
2. Mở popup → chọn **🧠 AI**, **🎯 Laya** hoặc **⚡ Bruteforce**.
3. Bấm **▶️ Bắt đầu giải Slide** — Extension tự trả lời, bấm `Kiểm tra` / `Câu tiếp theo` / `Trang sau`, hết bài thì **⏹️ Dừng lại**.

| Chế độ | Cách chọn đáp án | Tốc độ | Cần |
| --- | --- | --- | --- |
| 🧠 AI | LLM chọn 1 đáp án, sai thì thử tuần tự | chậm nhất (1–5s/câu) | API key hoặc Ollama |
| 🎯 Laya 🧪 | [Laya](https://github.com/NandhaKishorM/laya) chấm xác suất mọi đáp án trong 1 lượt → thử từ cao xuống thấp | nhanh (~vài chục–vài trăm ms/câu) | `laya-serve` chạy local |
| ⚡ Bruteforce | Thử A → B → C… | nhanh nhất | không cần gì |

> [!WARNING]
> **🎯 Laya đang ở giai đoạn thử nghiệm.** Laya là model phân loại (không phải LLM), nên hiểu kiến thức kém hơn nhiều so với AI.
> Trên 16 câu hỏi tiếng Việt tự soạn (CPU, `laya-multilingual`): đúng ngay lần đầu **8/16**, trung bình **1,75** lần thử/câu (đoán ngẫu nhiên: 2,5), ~60–90 ms/câu.
> Nghĩa là ít click sai hơn Bruteforce, nhưng chưa được kiểm chứng trên câu hỏi EDUX thật — kết quả có thể khác. Hãy báo lại nếu bạn dùng thử!

**Cài Laya (một lần, cần Python ≥ 3.10):**

```powershell
py -m venv laya-env
.\laya-env\Scripts\python.exe -m pip install "laya[serve]"
$env:LAYA_MODELS = "multilingual"; .\laya-env\Scripts\laya-serve.exe   # http://localhost:8000, lần đầu tải model (~1GB)
```

Rồi vào tab **⚙️ Cài đặt** → **Laya server** → **🔌 Kiểm tra**. Đổi địa chỉ nếu đặt `LAYA_PORT` khác; nếu đặt `LAYA_API_KEY` thì nhập key vào ô bên dưới.

### 3. Giải Bài tập (tab 📝 Bài tập)

Mở bài tập EDUX (hoặc bấm **🚀 Mở bài** trong popup) → tab **Bài tập**:

- **⚡ Tự động (API):** bấm **GIẢI BÀI TẬP BẰNG AI** → theo dõi 3 bước `Bắt đề → AI giải → Điền & Nộp`. Muốn đổi model thì bấm **⚙️ Đổi Model**.
- **💬 Chatbot (không cần key):**
  1. **Bước 1:** bấm **📋 Copy đề & xem trước** → đề được copy sẵn, có thể mở nhanh ChatGPT / Gemini / Claude ngay trong popup.
  2. **Bước 2:** dán đề vào chatbot, copy đáp án → về popup bấm **📥 Dán Clipboard** (lần đầu trình duyệt sẽ hỏi quyền clipboard → chọn **Cho phép**) hoặc `Ctrl+V` vào ô.
  3. **Bước 3:** bấm **✨ Bắt đầu điền bài tập**.
- **Thời gian chờ mỗi câu** (tab ⚙️ Cài đặt): khi tự điền, chờ sau mỗi câu rồi mới sang câu tiếp / nộp bài — **Cố định** (VD `5` giây) hoặc **Ngẫu nhiên** trong khoảng (VD `3`–`8` giây). Mặc định `0` = không chờ. Áp dụng cho cả chế độ Tự động (API) và Chatbot.
- Nút **🔄 Phiên mới** để xóa đáp án cũ khi làm bài khác.

### 4. Xem điểm (tab 📊 Điểm số)

- Trang `/student`: thanh tiến độ ngay trên từng thẻ môn học.
- Trang môn học: huy hiệu `🏆 X/10` hoặc `⚠️ Chưa làm` cạnh mỗi bài.
- Popup → tab **Điểm số** → **🔄 Quét lại** để xem bảng tổng hợp.

---

## 🗂 Cấu trúc repo

```text
EDUX-EXTENSION/          # Extension chính (duy nhất còn phát triển)
├── manifest.json        # Manifest V3, v2.6.0
├── background/          # Service worker: client AI, Laya, kiểm tra bản mới
├── content/             # Content script: giải Slide, Bài tập, theo dõi điểm
├── popup/               # Giao diện popup, mỗi tab một module
├── shared/              # Bảng provider AI và hằng số dùng chung
├── update.bat / update.ps1  # Cập nhật tại chỗ, giữ nguyên cấu hình
tests/                   # Unit test, golden test, smoke test (Playwright)
tools/                   # Đóng gói bản phát hành
legacy/                  # [ngừng hỗ trợ] script Python/Playwright cũ — xem legacy/README.md
```

Kiến trúc chi tiết và lệnh phát triển (`npm test`, `npm run smoke`...): xem [EDUX-EXTENSION/README.md](EDUX-EXTENSION/README.md#-kiến-trúc-cho-người-phát-triển).

---

> Tìm script Python/Playwright cũ? Chúng đã ngừng hỗ trợ và nằm trong [`legacy/`](legacy/).

---

## 📜 Miễn trừ trách nhiệm

_Công cụ phục vụ mục đích nghiên cứu và học tập. Người dùng tự chịu trách nhiệm khi sử dụng trên nền tảng EDUX._
