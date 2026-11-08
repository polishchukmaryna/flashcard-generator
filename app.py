import os
import gradio as gr
from groq import Groq


def generate_flashcard(text: str, api_key: str) -> str:
    if not api_key or not api_key.strip():
        return "Помилка: будь ласка, введіть API-ключ Groq."

    if not text or not text.strip():
        return "Помилка: будь ласка, введіть текст для створення картки."

    try:
        client = Groq(api_key=api_key.strip())

        system_prompt = (
            "Ти - генератор навчальних карток.\n"
            "На основі наданого тексту створи рівно одну навчальну картку українською мовою.\n\n"
            "Вимоги:\n"
            "1. Створи одне запитання за найважливішою інформацією з тексту.\n"
            "2. Надай одну однозначну правильну відповідь.\n"
            "3. Відповідь повинна ґрунтуватися виключно на наданому тексті.\n"
            "4. Не додавай інформацію, якої немає в тексті.\n"
            "5. Не пиши вступів, привітань або додаткових пояснень.\n"
            "6. Використовуй такий формат:\n\n"
            "Запитання: ...\n"
            "Відповідь: ...\n\n"
            "Якщо інформації недостатньо для створення однозначної картки, "
            "повідом про це."
        )

        chat_completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": text.strip()
                }
            ],
            temperature=0.2
        )

        result = chat_completion.choices[0].message.content

        if not result:
            return "Не вдалося отримати відповідь від мовної моделі."

        return result.strip()

    except Exception as e:
        return f"Помилка API: {str(e)}"


demo = gr.Interface(
    fn=generate_flashcard,
    inputs=[
        gr.Textbox(
            label="Текст для створення картки",
            placeholder="Вставте сюди текст...",
            lines=8
        ),
        gr.Textbox(
            label="API-ключ Groq",
            placeholder="Введіть свій API-ключ...",
            type="password"
        )
    ],
    outputs=gr.Textbox(
        label="Створена навчальна картка",
        lines=6
    ),
    title="Flashcard Generator",
    description=(
        "Генератор навчальних карток на основі тексту "
        "за допомогою мовної моделі штучного інтелекту. "
        "Введіть текст та API-ключ Groq, щоб отримати "
        "одне запитання та одну правильну відповідь."
    )
)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True
    )