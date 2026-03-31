import requests
import json
import base64
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")

PROMPT = """
    You are a strict OCR extraction system for handwritten university exam papers.

    Your task is to extract all visible problems and student solutions from the exam pages and return ONLY valid JSON.

    IMPORTANT RULES:
    1. Do NOT summarize.
    2. Do NOT explain.
    3. Do NOT interpret the student's reasoning.
    4. Do NOT correct mathematics.
    5. Do NOT simplify formulas.
    6. Copy all visible expressions as faithfully as possible.
    7. If some symbols are unclear, keep the partial expression and mention the issue in notes.
    8. Extract every visible task.
    9. Preserve the original order of tasks.
    10. Return ONLY JSON.

    The required JSON structure is:
    {
      "tasks": [
        {
          "question_id": "string",
          "text": [
            "full task text as written in the exam"
          ],
          "student_solution": {
            "steps": [
              "student's reasoning steps"
            ],
            "calculation": [
              "mathematical expressions and transformations"
            ],
            "result": "final written answer",
            "notes": [],
            "source_pages": [1]
          }
        }
      ]
    }

    Field rules:
    - question_id: the task number exactly as written in the exam
    - text: full text of the current task only
    - steps: logical steps of the student's solution
    - calculation: mathematical calculations and formulas
    - result: final written answer
    - notes: OCR uncertainty notes such as "illegible symbol" or "crossed out fragment"
    - source_pages: page numbers where this task appears

    If there are no notes, return an empty array.
    If the final answer is missing, return an empty string.
    Return only valid JSON. """


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# --- 2. GPT-4o-mini ҮШІН (Әр бетті жеке өңдеу - result-тар жоғалмауы үшін) ---
def get_gpt_response_full(image_paths, output_filename):
    print(f"\n🚀 [GPT-4o-mini] топталған беттерді өңдеп жатыр...")
    all_tasks = []

    # Суреттерді 2-ден топтаймыз (бұл контекстті сақтауға көмектеседі)
    chunks = [image_paths[i:i + 2] for i in range(0, len(image_paths), 2)]

    for chunk in chunks:
        print(f"   📄 Өңделуде: {chunk}...")
        content = [{"type": "text", "text": PROMPT}]
        for path in chunk:
            base64_img = encode_image(path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
            })

        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            data=json.dumps({
                "model": "openai/gpt-4o-mini",
                "messages": [{"role": "user", "content": content}],
                "temperature": 0
            })
        )

        try:
            res_data = response.json()
            content_str = res_data['choices'][0]['message']['content'].replace('```json', '').replace('```', '').strip()
            data = json.loads(content_str)
            all_tasks.extend(data.get("tasks", []))
        except Exception as e:
            print(f"❌ Қате: {e}")

    # Соңында нәтижені біріктіру
    final_json = {"tasks": all_tasks}
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(final_json, f, ensure_ascii=False, indent=2)


# --- 3. QWEN ҮШІН (Барлық суретті бірден өңдеу) ---
def get_qwen_response(image_paths, output_filename):
    print(f"\n🚀 [Qwen 2.5 VL] топталған беттерді өңдеп жатыр...")
    all_tasks = []

    # Суреттерді 2-ден топтаймыз (Лимиттен аспау үшін)
    chunks = [image_paths[i:i + 2] for i in range(0, len(image_paths), 2)]

    for chunk in chunks:
        print(f"   📄 Өңделуде: {chunk}...")
        content = [{"type": "text", "text": PROMPT}]
        for path in chunk:
            base64_img = encode_image(path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
            })

        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                data=json.dumps({
                    "model": "qwen/qwen2.5-vl-72b-instruct",
                    "messages": [{"role": "user", "content": content}],
                    "temperature": 0
                }),
                timeout=300
            )

            result = response.json()

            if 'choices' in result:
                raw_content = result['choices'][0]['message']['content'].replace('```json', '').replace('```',
                                                                                                        '').strip()
                try:
                    data = json.loads(raw_content)
                    all_tasks.extend(data.get("tasks", []))
                except:
                    # Егер JSON емес, мәтін келсе, солай сақтаймыз
                    print(f"⚠️ {chunk} үшін жауап таза JSON емес")
            else:
                # Егер 'choices' жоқ болса, қате мәтінін көреміз
                print(f"❌ Qwen қатесі: {json.dumps(result, indent=2)}")

        except Exception as e:
            print(f"❌ Сұраныс жіберуде қате: {e}")

    # Барлық жиналған есептерді бір файлға сақтау
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump({"tasks": all_tasks}, f, ensure_ascii=False, indent=2)
    print(f"✅ Qwen нәтижесі сақталды: {output_filename}")


# --- 4. DEEPSEEK ҮШІН (Relay әдісі - суретсіз, тек мәтінмен) ---
def get_deepseek_relay(qwen_json_file, output_filename):
    print(f"\n🚀 [DeepSeek-R1] Qwen деректерін өңдеп жатыр...")

    try:
        with open(qwen_json_file, "r", encoding="utf-8") as f:
            qwen_text = f.read()

        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            data=json.dumps({
                "model": "deepseek/deepseek-r1",
                "messages": [
                    {"role": "system",
                     "content": "You are a professional editor. Clean up the provided JSON OCR data. Return ONLY the JSON."},
                    {"role": "user", "content": f"Input JSON: {qwen_text}"}
                ],
                "temperature": 0.5,  # Reasoning модельдер үшін 0.5 - 0.7 жақсырақ
                "max_tokens": 4000  # Жауап толық болуы үшін
            }),
            timeout=300  # DeepSeek ұзақ ойланады, сондықтан күту уақытын создық
        )

        result = response.json()

        # Егер жауап дұрыс келсе:
        if 'choices' in result:
            content = result['choices'][0]['message']['content'].replace('```json', '').replace('```', '').strip()
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"✅ DeepSeek нәтижесі сақталды: {output_filename}")
        else:
            # Егер қате болса, нақты не қате екенін принтеп шығарамыз
            print(f"❌ DeepSeek қатесі: {json.dumps(result, indent=2)}")

    except Exception as e:
        print(f"❌ DeepSeek функциясында қате шықты: {e}")


# --- ІСКЕ ҚОСУ ---
exam_pages = ["72_1.jpg", "72_2.jpg", "72_3.jpg", "72_4.jpg", "72_5.jpg"]

get_qwen_response(exam_pages, "qwen.json")
get_gpt_response_full(exam_pages, "gpt.json")
get_deepseek_relay("qwen.json", "deepseek.json")