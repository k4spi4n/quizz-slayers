import pyautogui
import easyocr
import ollama
import time
import os
import sys
import keyboard
import numpy as np
import re
import hashlib
from fuzzywuzzy import fuzz
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
BUTTON_IMAGE_PATH = r'..\buttons\show_question_button.png'
CHECK_BUTTON_PATH = r'..\buttons\check_answer_button.png'
NEXT_BUTTON_PATH = r'..\buttons\next_page_button.png'

OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'hf.co/arcee-ai/Arcee-VyLinh-GGUF:Q8_0')
LANGUAGES = ['vi']
CONFIDENCE_THRESHOLD = 0.8  
FUZZY_MATCH_THRESHOLD = 80  
OLLAMA_KEEP_ALIVE_SECONDS = 600
SYSTEM_PROMPT = """Bạn là trợ lí thông minh chuyên trả lời câu hỏi. Yêu cầu:
1. Xác định câu trả lời đúng. 
2. Trả về CHÍNH XÁC nguyên văn dòng text của đáp án đó xuất hiện trên màn hình (gồm cả ký tự A/B/C/D và đáp án). 
3. KHÔNG giải thích, KHÔNG thêm lời dẫn. 
Ví dụ: Nếu đáp án đúng là 'C. Hà Nội', hãy in ra chính xác: C. Hà Nội.
"""
ANSWER_LINE_PATTERN = re.compile(r"^[A-Da-d][\)\.]?\s+\S+")

# Ollama options tuned for lower latency without changing model
OLLAMA_OPTIONS = {
    "temperature": 0.1,
    "top_p": 0.3,
    "num_predict": 64,
}

# PyAutoGUI Safety
pyautogui.FAILSAFE = True

def get_center_of_box(box):
    """
    Calculates the center (x, y) of an EasyOCR bounding box.
    Box format: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    """
    try:
        (tl, tr, br, bl) = box
        center_x = int((tl[0] + br[0]) / 2)
        center_y = int((tl[1] + br[1]) / 2)
        return center_x, center_y
    except Exception as e:
        print(f"[!] Error calculating box center: {e}")
        return None

def find_best_match_location(target_text, ocr_results):
    """
    Scans OCR results to find the text bounding box that best matches 
    the target_text using fuzzy string matching.
    """
    best_score = 0
    best_loc = None
    best_text_found = ""

    target_norm = target_text.lower().strip()
    print(f"[*] Searching for text on screen: '{target_norm}'")

    for (box, text, prob) in ocr_results:
        text_norm = text.lower().strip()

        score_ratio = fuzz.ratio(target_norm, text_norm)
        score_partial = fuzz.partial_ratio(target_norm, text_norm)

        final_score = max(score_ratio, score_partial)

        if len(text_norm) < 2:
            final_score = 0

        if final_score > best_score:
            best_score = final_score
            best_loc = get_center_of_box(box)
            best_text_found = text

    if best_score >= FUZZY_MATCH_THRESHOLD:
        print(f"[*] MATCH FOUND: '{best_text_found}' (Score: {best_score})")
        return best_loc
    else:
        print(f"[!] No strong match found. Best candidate: '{best_text_found}' (Score: {best_score} < {FUZZY_MATCH_THRESHOLD})")
        return None

def find_and_click_image(image_path, description="image", retries=1):
    """
    Attempts to locate and click an image on screen.
    """
    print(f"[*] Looking for {description}...")
    
    for i in range(retries + 1):
        try:
            location = pyautogui.locateCenterOnScreen(image_path, confidence=CONFIDENCE_THRESHOLD, grayscale=True)
            if location:
                print(f"[*] Found {description}. Clicking...")
                pyautogui.click(location)
                return True
            else:
                if i < retries:
                    time.sleep(0.5)
        except pyautogui.ImageNotFoundException:
            pass 
        except Exception as e:
            print(f"[!] Error searching for image: {e}")
    
    print(f"[!] Could not find {description}.")
    return False

def check_ollama_connection():
    print("[*] Checking Ollama connection...")
    try:
        models = ollama.list()
        print("[*] Ollama is online.")
        return True
    except Exception as e:
        print(f"[!] Could not connect to Ollama. Is it running? Error: {e}")
        return False

def extract_relevant_text(ocr_results):
    """
    Extracts a compact question + answers block to reduce prompt size.
    Falls back to full OCR text if no answer-like lines are found.
    """
    lines = [res[1].strip() for res in ocr_results if res[1].strip()]
    if not lines:
        return "", []

    answer_indices = [i for i, line in enumerate(lines) if ANSWER_LINE_PATTERN.match(line)]
    if not answer_indices:
        return ". ".join(lines), lines

    first_ans = answer_indices[0]
    last_ans = answer_indices[-1]
    question_lines = lines[:first_ans]
    answer_lines = lines[first_ans:last_ans + 1]

    compact_lines = question_lines + answer_lines
    return "\n".join(compact_lines).strip(), lines

def solve_quiz(reader, answer_cache):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_show_path = os.path.join(script_dir, BUTTON_IMAGE_PATH)
    abs_check_path = os.path.join(script_dir, CHECK_BUTTON_PATH)
    abs_next_path = os.path.join(script_dir, NEXT_BUTTON_PATH)

    # 1. Click 'Show Question'
    # We try this, but if not found, we assume the question might already be visible
    if find_and_click_image(abs_show_path, "Show Question Button"):
        time.sleep(0.8) 

    # 2. Capture Screen
    print("[*] Capturing screen...")
    try:
        screenshot = pyautogui.screenshot()
        screenshot_np = np.array(screenshot)
    except Exception as e:
        print(f"[!] Screenshot failed: {e}")
        return

    # 3. OCR
    print("[*] Reading text (OCR)...")
    ocr_results = reader.readtext(screenshot_np, detail=1)
    
    compact_text, full_text_lines = extract_relevant_text(ocr_results)
    if not compact_text:
        print("[!] No text detected on screen.")
        return

    print(f"\n--- Screen Content (Compact) ---\n{compact_text}\n----------------------")

    cache_key = hashlib.sha1(compact_text.encode("utf-8")).hexdigest()
    if cache_key in answer_cache:
        ai_answer_text = answer_cache[cache_key]
        print(f"\n=== AI ANSWER (CACHED): {ai_answer_text} ===\n")
    else:
        # 4. Query AI
        print(f"[*] Sending to AI ({OLLAMA_MODEL})...")
        prompt = (
            "Dưới đây là nội dung câu hỏi và các đáp án có thể chọn trên màn hình:\n"
            f"{compact_text}\n"
            "CÂU TRẢ LỜI CỦA BẠN: "
        )

        try:
            response = ollama.chat(model=OLLAMA_MODEL, messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': prompt}
            ], options=OLLAMA_OPTIONS, keep_alive=OLLAMA_KEEP_ALIVE_SECONDS)
            
            ai_answer_text = response['message']['content'].strip()
            answer_cache[cache_key] = ai_answer_text
            
            print(f"\n=== AI ANSWER: {ai_answer_text} ===\n")
        except Exception as e:
            print(f"[!] Error during AI/Click sequence: {e}")
            return

    # 5. Click Answer
    try:
        click_coords = find_best_match_location(ai_answer_text, ocr_results)

        if click_coords:
            print(f"[*] Clicking answer at {click_coords}...")
            pyautogui.moveTo(click_coords[0], click_coords[1])
            pyautogui.click()
            
            # 6. Click Check & Next
            time.sleep(0.5)
            find_and_click_image(abs_check_path, "Check Answer Button")
            
            time.sleep(0.5) 
            pyautogui.click()
            
        else:
            print(f"[!] Could not locate answer '{ai_answer_text}' on screen.")
    except Exception as e:
        print(f"[!] Error during AI/Click sequence: {e}")

def main():
    print("--- QUIZ SLAYER BOT ---")
    
    if not check_ollama_connection():
        print("Please start Ollama and try again.")
        return

    print("Initializing EasyOCR... (Loading models)")
    try:
        reader = easyocr.Reader(LANGUAGES, gpu=True)
        print("EasyOCR initialized successfully.")
    except Exception as e:
        print(f"[!] EasyOCR Error: {e}")
        print("Try installing pytorch/cuda or set gpu=False in code.")
        return

    print("\n" + "="*50)
    print(" BOT READY TO SLAY ")
    print(" [F]   -> Solve ONE question")
    print(" [G]   -> Auto-Solve Mode (Loop)")
    print(" [Esc] -> Quit")
    print("="*50 + "\n")

    answer_cache = {}

    # Main Loop
    while True:
        try:
            # --- EXIT ---
            if keyboard.is_pressed('esc'):
                print("\n[!] Exiting...")
                break
            
            # --- MANUAL MODE (F) ---
            if keyboard.is_pressed('f'):
                print("\n>>> [Manual] 'F' Triggered")
                while keyboard.is_pressed('f'): time.sleep(0.1) 
                
                solve_quiz(reader, answer_cache)
                print("\n[*] Manual sequence finished.")

            # --- AUTO MODE (G) ---
            if keyboard.is_pressed('g'):
                print("\n>>> [AUTO MODE] STARTED. Press 'Ctrl + C' to stop.")
                while keyboard.is_pressed('g'): time.sleep(0.1) 
                
                while True:
                    solve_quiz(reader, answer_cache)
                    time.sleep(0.5)

            time.sleep(0.05)
            
        except KeyboardInterrupt:
            print("\n[!] User interrupted.")
            break
        except Exception as e:
            print(f"[!] Unexpected error in main loop: {e}")

if __name__ == "__main__":
    main()