import requests
import json
import base64

API_KEY = "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"

# nurrr api key  | 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
# nurbooo api key | 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
# myKey api key  | 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

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


def get_ai_response_multi_images(model_name, image_paths, output_filename):
    print(f"\n🚀 [{model_name}] моделіне сұраныс жіберілуде...")

    content = [{"type": "text", "text": PROMPT}]

    for path in image_paths:
        base64_img = encode_image(path)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
        })

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            data=json.dumps({
                "model": model_name,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0
            }),
            timeout=240
        )

        result = response.json()

        if 'choices' in result:
            raw_content = result['choices'][0]['message']['content']

            clean_json_str = raw_content.replace('```json', '').replace('```', '').strip()

            try:
                parsed_json = json.loads(clean_json_str)
                with open(output_filename, "w", encoding="utf-8") as f:
                    json.dump(parsed_json, f, ensure_ascii=False, indent=2)
                print(f"✅ Сәтті сақталды: {output_filename}")
            except json.JSONDecodeError:
                with open(output_filename, "w", encoding="utf-8") as f:
                    f.write(clean_json_str)
                print(f"⚠️ Ескерту: Жауап таза JSON емес, бірақ мәтін ретінде сақталды: {output_filename}")

        else:
            print(f"❌ Қате орын алды ({model_name}):", result)

    except Exception as e:
        print(f"❌ Сервермен байланыс үзілді немесе қате шықты ({model_name}): {e}")


exam_pages = ["72_1.jpg", "72_2.jpg", "72_3.jpg", "72_4.jpg", "72_5.jpg"]

get_ai_response_multi_images("openai/gpt-4o-mini", exam_pages, "gpt.json")
get_ai_response_multi_images("deepseek/deepseek-r1", exam_pages, "deepseek.json")
get_ai_response_multi_images("qwen/qwen2.5-vl-72b-instruct", exam_pages, "../qwen.json")