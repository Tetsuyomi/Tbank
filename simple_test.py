import torch
from transformers import RobertaTokenizerFast, RobertaForQuestionAnswering

# Пример тестовых данных
test_data = [
    {
        "question": "Кто написал 'Войну и мир'?",
        "context": "Роман 'Война и мир' был написан Львом Николаевичем Толстым.",
        "answer": "Лев Николаевич Толстой"
    },
    {
        "question": "Что такое машинное обучение?",
        "context": "Машинное обучение — это раздел искусственного интеллекта.",
        "answer": "раздел искусственного интеллекта"
    },
    {
        "question": "Кто написал 'Мастера и Маргариту'?",
        "context": "Роман 'Мастер и Маргарита' был написан Михаилом Булгаковым.",
        "answer": "Михаил Булгаков"
    }
]


#Функция для получения ответа модели
def answer_question(model, tokenizer, question, context):
    #Токенизация данных
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt", max_length=512, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    #Получение предсказаний модели
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

    #Определение стартовой и конечной позиций ответа
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores) + 1

    #Преобразование индексов в текст
    answer = tokenizer.decode(input_ids[0][answer_start:answer_end], skip_special_tokens=True)
    return answer


#Используйте GPU если доступно
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Модели которые нужно проверить
checkpoints = [
    "model_roberta/checkpoint-11332",
    "model_roberta/checkpoint-16998",
    "model_roberta/checkpoint-5666"
]

#Проверка точности для каждой модели
for checkpoint in checkpoints:
    print(f"Проверка точности модели: {checkpoint}")

    #Загрузка токенизатора и модели
    tokenizer = RobertaTokenizerFast.from_pretrained(checkpoint)
    model = RobertaForQuestionAnswering.from_pretrained(checkpoint).to(device)

    #Проверка на каждом примере
    for example in test_data:
        question = example["question"]
        context = example["context"]
        correct_answer = example["answer"]

        #Получение ответа модели
        predicted_answer = answer_question(model, tokenizer, question, context)

        # Вывод результатов
        print(f"Вопрос: {question}")
        print(f"Ответ модели: {predicted_answer}")
        print(f"Ожидаемый ответ: {correct_answer}")
        print("-" * 50)

    print("=" * 50)
