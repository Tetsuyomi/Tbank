from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram import F
from transformers import BertTokenizerFast, BertForQuestionAnswering
import torch
import asyncio
from config import TOKEN

# # Токен бота
# API_TOKEN = TOKEN
#
# # Инициализация бота и диспетчера
# bot = Bot(token=API_TOKEN)
# dp = Dispatcher()
#
# # Загрузка модели и токенизатора
# # tokenizer = RobertaTokenizerFast.from_pretrained('./Tbank_intr/model_roberta/checkpoint-16998')
# # model = RobertaForQuestionAnswering.from_pretrained('./Tbank_intr/model_roberta/checkpoint-16998').to('cuda')
# tokenizer = BertTokenizerFast.from_pretrained(r'D:\Python\PycharmProjects\neiro\Tbank_intr\models_first_epoch\checkpoint-4000')
# model = BertForQuestionAnswering.from_pretrained(r'D:\Python\PycharmProjects\neiro\Tbank_intr\models_first_epoch\checkpoint-4000').to('cuda')
#
#
# # Состояние чата для контекста
# context_storage = {}
# active_users = set()  # Множество активных пользователей, выполнивших команду /start
#
#
# @dp.message(Command('start'))
# async def send_welcome(message: types.Message):
#     chat_id = message.chat.id
#
#     # Добавляем пользователя в список активных
#     active_users.add(chat_id)
#
#     # Приветственное сообщение
#     await message.answer("Привет! Это бот для ответов на вопросы.")
#     # Просим пользователя ввести контекст
#     await message.answer("Введите контекст, на основе которого я буду отвечать на вопросы:")
#
#
# @dp.message()
# async def handle_messages(message: types.Message):
#     chat_id = message.chat.id
#
#     # Если контекст еще не введен
#     if chat_id not in context_storage:
#         context_storage[chat_id] = message.text
#         await message.answer("Контекст сохранён. Теперь задай мне вопрос.")
#     else:
#         context = context_storage[chat_id]
#         question = message.text
#
#         # Предобработка и получение ответа от модели
#         inputs = tokenizer.encode_plus(question, context, return_tensors="pt", max_length=512, truncation=True)
#         input_ids = inputs["input_ids"].to('cuda')
#         attention_mask = inputs["attention_mask"].to('cuda')
#
#         with torch.no_grad():
#             outputs = model(input_ids, attention_mask=attention_mask)
#             start_scores, end_scores = outputs.start_logits, outputs.end_logits
#             answer_start = torch.argmax(start_scores)
#             answer_end = torch.argmax(end_scores) + 1
#             answer = tokenizer.decode(input_ids[0][answer_start:answer_end], skip_special_tokens=True)
#
#         await message.answer(f"Ответ: {answer}")
#         del context_storage[chat_id]  # Очистка контекста после ответа
#
#
# async def main():
#     # Запуск бота
#     await bot.delete_webhook(drop_pending_updates=True)
#     await dp.start_polling(bot)
#
# if __name__ == '__main__':
#     asyncio.run(main())


#Токен
API_TOKEN = TOKEN

#Инициализация бота и диспетчера
bot = Bot(token=API_TOKEN)
dp = Dispatcher()

#Загрузка модели и токенизатора
tokenizer = BertTokenizerFast.from_pretrained(r'D:\Python\PycharmProjects\neiro\Tbank_intr\models_first_epoch\checkpoint-4000')
model = BertForQuestionAnswering.from_pretrained(r'D:\Python\PycharmProjects\neiro\Tbank_intr\models_first_epoch\checkpoint-4000').to('cuda')

#Состояние чата
context_storage = {}
active_users = set()  # Множество активных пользователей, выполнивших команду /start

@dp.message(Command('start'))
async def send_welcome(message: types.Message):
    chat_id = message.chat.id

    active_users.add(chat_id)

    # Приветственное сообщение
    await message.answer("Привет! Это бот для ответов на вопросы.")
    # Просим пользователя ввести контекст
    await message.answer("Введите контекст, на основе которого я буду отвечать на вопросы:")

# Обработчик для неактивных пользователей (до команды /start)
@dp.message(lambda message: message.chat.id not in active_users)
async def handle_inactive_user(message: types.Message):
    await message.answer("Пожалуйста, введите команду /start для начала работы.")

# Основной обработчик сообщений для активных пользователей
@dp.message(lambda message: message.chat.id in active_users)
async def handle_messages(message: types.Message):
    chat_id = message.chat.id

    # Если контекст еще не введен
    if chat_id not in context_storage:
        context_storage[chat_id] = message.text
        await message.answer("Контекст сохранён. Теперь задай мне вопрос.")
    else:
        context = context_storage[chat_id]
        question = message.text

        # Предобработка и получение ответа от модели
        inputs = tokenizer.encode_plus(question, context, return_tensors="pt", max_length=512, truncation=True)
        input_ids = inputs["input_ids"].to('cuda')
        attention_mask = inputs["attention_mask"].to('cuda')

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            start_scores, end_scores = outputs.start_logits, outputs.end_logits
            answer_start = torch.argmax(start_scores)
            answer_end = torch.argmax(end_scores) + 1
            answer = tokenizer.decode(input_ids[0][answer_start:answer_end], skip_special_tokens=True)

        await message.answer(f"Ответ: {answer}")
        del context_storage[chat_id]  # Очистка контекста после ответа

    # Запуск бота
async def main():
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())

