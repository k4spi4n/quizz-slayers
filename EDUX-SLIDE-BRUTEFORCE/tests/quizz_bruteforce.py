import os
import random

from playwright.sync_api import Page

LOGIN_URL = "https://edux.cmcu.edu.vn/login"
ENV_PATH = os.path.join(os.path.dirname(__file__), "..", ".env")


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


def ensure_login_env() -> tuple[str, str]:
    load_env_file()
    email = os.environ.get("EDUX_EMAIL", "").strip()
    password = os.environ.get("EDUX_PASSWORD", "").strip()

    if not email:
        email = input("Enter EDUX email: ").strip()
    if not password:
        password = input("Enter EDUX password: ").strip()

    os.makedirs(os.path.dirname(ENV_PATH), exist_ok=True)
    with open(ENV_PATH, "w", encoding="utf-8") as env_file:
        env_file.write(f"EDUX_EMAIL={email}\n")
        env_file.write(f"EDUX_PASSWORD={password}\n")

    os.environ["EDUX_EMAIL"] = email
    os.environ["EDUX_PASSWORD"] = password
    return email, password


def extract_answer_texts(answers_locator) -> list[str]:
        return answers_locator.evaluate_all(
                """
                nodes => nodes.map(node => {
                    const letter = node.querySelector('span.font-bold')?.innerText?.trim() || '';
                    const text = node.querySelector('div.prose p')?.innerText?.trim() || '';
                    return `${letter} ${text}`.trim();
                })
                """
        )


def test_wait_for_user_login(page: Page) -> None:
    email, password = ensure_login_env()
    page.goto(LOGIN_URL, wait_until="domcontentloaded")
    page.locator("#email").fill(email)
    page.locator("#password").fill(password)
    page.locator("#password").press("Enter")
    print("\n[INFO] Auto-login attempted. If needed, finish any extra steps in the browser.")
    print("[INFO] After you reach the quiz screen, press Enter here to click 'Tra loi cau hoi'.\n")
    input()

    wrong_answers: dict[str, set[str]] = {}
    question_answer_cache: dict[str, list[str]] = {}

    no_question_button = page.get_by_role("button", name="Không có câu hỏi")
    answer_button = page.get_by_role("button", name="Trả lời trên lớp")
    check_button = page.get_by_role("button", name="Kiểm tra")
    next_button = page.get_by_role("button", name="Câu tiếp theo")
    retry_button = page.get_by_role("button", name="Thử lại")
    next_page_button = page.get_by_role("button", name="Trang sau")
    question_locator = page.locator("p.my-3.text-gray-800.leading-relaxed").first
    answers_locator = page.locator(
        "div.flex.items-center.space-x-6.p-8.rounded-xl.border-2.transition-colors.cursor-pointer.min-h-\\[80px\\]"
    )

    while not page.is_closed():
        if no_question_button.is_visible():
            next_page_button.click()
            print("[INFO] No question on this slide. Clicked 'Trang sau'.")
            page.wait_for_load_state("domcontentloaded")
            page.wait_for_timeout(200)
            continue

        if not question_locator.is_visible():
            answer_button.wait_for(state="visible")
            answer_button.click()

        try:
            question_locator.wait_for(state="visible", timeout=10000)
        except Exception:
            print("[WARN] Question not visible yet. Retrying loop.")
            page.wait_for_timeout(200)
            continue

        question_text = question_locator.inner_text().strip()
        print(f"[INFO] Question: {question_text}")

        answer_texts = question_answer_cache.get(question_text)
        if answer_texts is None:
            try:
                answers_locator.first.wait_for(state="visible", timeout=10000)
            except Exception:
                print("[WARN] Answers not visible yet. Retrying loop.")
                page.wait_for_timeout(200)
                continue

            answer_texts = extract_answer_texts(answers_locator)
            question_answer_cache[question_text] = answer_texts

        answer_count = len(answer_texts)
        print(f"[INFO] Answers found: {answer_count}")

        target_answer = os.environ.get("AUTO_ANSWER_TEXT", "").strip()
        clicked_answer = False
        chosen_answer_text = ""
        if target_answer:
            print(f"[INFO] Auto-answer target: {target_answer}")
            match = answers_locator.filter(has_text=target_answer).first
            match.click()
            clicked_answer = True
            chosen_answer_text = target_answer
        elif answer_count > 0:
            tried_for_question = wrong_answers.get(question_text, set())
            next_index = next(
                (i for i, text in enumerate(answer_texts) if text not in tried_for_question),
                None,
            )
            if next_index is None:
                tried_for_question.clear()
                next_index = 0

            chosen_answer_text = answer_texts[next_index]
            print(f"[INFO] AUTO_ANSWER_TEXT not set. Pick: {chosen_answer_text}")
            answers_locator.nth(next_index).click()
            clicked_answer = True
        else:
            print("[WARN] No answers available to click.")

        if clicked_answer:
            check_button.click()
            print("[INFO] Clicked 'Kiem tra' button.")

            try:
                page.wait_for_function(
                    """
                    () => {
                      const labels = ['Trang sau', 'Câu tiếp theo', 'Thử lại'];
                      return labels.some(label => {
                        const btn = Array.from(document.querySelectorAll('button'))
                          .find(b => (b.textContent || '').trim() === label);
                        return btn && !btn.disabled && btn.offsetParent !== null;
                      });
                    }
                    """,
                    timeout=10000,
                )

                if next_page_button.is_visible():
                    next_page_button.click()
                    print("[INFO] Clicked 'Trang sau' button.")
                elif next_button.is_visible():
                    next_button.click()
                    print("[INFO] Clicked 'Cau tiep theo' button.")
                elif retry_button.is_visible():
                    retry_button.click()
                    if chosen_answer_text:
                        wrong_answers.setdefault(question_text, set()).add(chosen_answer_text)
                        print(f"[INFO] Marked wrong answer: {chosen_answer_text}")
                    print("[INFO] Clicked 'Thu lai' button.")
                else:
                    print("[WARN] Follow-up buttons not visible after wait.")
            except Exception:
                print("[WARN] No follow-up button appeared.")

        page.wait_for_timeout(200)

    # Keep this test non-failing while we are still wiring selectors.
    print(f"[INFO] Current URL after click: {page.url}")
    print("[INFO] Browser will stay open. Close the browser window to finish.")
    page.wait_for_event("close")
