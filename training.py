import os
import torch
from transformers import BertTokenizerFast, BertForQuestionAnswering, TrainingArguments, Trainer
from datasets import load_dataset

#Путь к базовой папке
base_dir = "training"

#Создание основной папки
os.makedirs(base_dir, exist_ok=True)

#Папки для модели
model_dir = os.path.join(base_dir, "dataset")
metrics_dir = os.path.join(base_dir, "metrics")

#Создание папок для сохранения модели
os.makedirs(model_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)

#Загрузка датасета Sberquad
dataset = load_dataset("sberquad")

#Загрузка токенизатора и модели
tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")
model = BertForQuestionAnswering.from_pretrained("bert-base-multilingual-cased")

#Токенизация данных
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=512,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i in range(len(answers)):
        start_char = answers[i]['answer_start'][0]
        end_char = start_char + len(answers[i]['text'][0])

        sequence_ids = inputs.sequence_ids(i)

        context_start = sequence_ids.index(1)
        context_end = len(sequence_ids) - sequence_ids[::-1].index(1) - 1

        offsets = inputs['offset_mapping'][i]

        start_token_idx = context_start
        end_token_idx = context_end
        for idx, (start, end) in enumerate(offsets):
            if start <= start_char and end >= start_char:
                start_token_idx = idx
            if start <= end_char and end >= end_char:
                end_token_idx = idx
                break

        start_positions.append(start_token_idx)
        end_positions.append(end_token_idx)

    inputs['start_positions'] = start_positions
    inputs['end_positions'] = end_positions
    return inputs

train_dataset = dataset["train"].map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

training_args = TrainingArguments(
    output_dir=model_dir,
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    logging_dir=metrics_dir,
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model(model_dir)

print("Дообучение завершено. Модель и метрики сохранены.")
