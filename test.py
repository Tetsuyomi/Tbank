import os
import torch
from transformers import RobertaTokenizerFast, RobertaForQuestionAnswering

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoints_dir = 'model_roberta/'  # Папка с чекпоинтами
output_dir = 'test4/'  # Папка для сохранения результатов

checkpoints = [f for f in os.listdir(checkpoints_dir) if f.startswith("checkpoint")]

os.makedirs(output_dir, exist_ok=True)

test_data = [
    {
        "question": "Что такое машинное обучение?",
        "context": """
        Машинное обучение — это раздел искусственного интеллекта, который занимается разработкой алгоритмов и моделей, способных обучаться на данных.
        Такие модели позволяют системам автоматически улучшаться на основе опыта без явного программирования.
        """,
        "answer": "раздел искусственного интеллекта"
    },
    {
        "question": "Что изучает физика?",
        "context": """
        Физика — это наука, которая изучает общие законы природы, включая явления движения, взаимодействия и строения материи.
        """,
        "answer": "наука"
    },
    {
        "question": "Кто написал 'Войну и мир'?",
        "context": """
        Роман 'Война и мир' был написан русским писателем Львом Николаевичем Толстым и является одним из величайших произведений мировой литературы.
        """,
        "answer": "Лев Николаевич Толстой"
    },
    {
        "question": "Что такое нейронная сеть?",
        "context": """
        Нейронная сеть — это модель машинного обучения, которая вдохновлена структурой и работой биологических нейронов в человеческом мозге.
        Она используется для решения задач классификации, регрессии и других сложных вычислительных задач.
        """,
        "answer": "модель машинного обучения"
    },
    {
        "question": "Что такое искусственный интеллект?",
        "context": """
        Искусственный интеллект — это область компьютерной науки, которая занимается созданием систем, способных выполнять задачи, требующие человеческого интеллекта.
        Это включает задачи такие как распознавание речи, визуальное восприятие, принятие решений и перевод текста.
        """,
        "answer": "область компьютерной науки"
    },
    {
        "question": "Кто написал 'Мастера и Маргариту'?",
        "context": """
        Роман 'Мастер и Маргарита' был написан Михаилом Афанасьевичем Булгаковым и считается одним из самых известных произведений русской литературы 20 века.
        """,
        "answer": "Михаил Афанасьевич Булгаков"
    },
    {
        "question": "Что такое квантовая механика?",
        "context": """
        Квантовая механика — это раздел физики, изучающий поведение материи и энергии на микроскопическом уровне, включая взаимодействие атомов и субатомных частиц.
        """,
        "answer": "раздел физики"
    },
    {
        "question": "Что такое глобальное потепление?",
        "context": """
        Глобальное потепление — это процесс долгосрочного повышения температуры Земли, вызванный деятельностью человека, такой как сжигание ископаемого топлива и вырубка лесов.
        """,
        "answer": "процесс долгосрочного повышения температуры"
    },
    {
        "question": "Кто написал 'Преступление и наказание'?",
        "context": """
        Роман 'Преступление и наказание' был написан русским писателем Фёдором Михайловичем Достоевским в 1866 году.
        """,
        "answer": "Фёдор Михайлович Достоевский"
    },
    {
        "question": "Что такое генетика?",
        "context": """
        Генетика — это наука, изучающая наследственность и изменчивость организмов. Она изучает, как передаются гены и как они влияют на развитие и функции организмов.
        """,
        "answer": "наука, изучающая наследственность и изменчивость"
    },
    {
        "question": "Что такое электричество?",
        "context": """
        Электричество — это форма энергии, возникающая в результате движения электрических зарядов, таких как электроны.
        Электричество используется для питания различных приборов и устройств.
        """,
        "answer": "форма энергии"
    },
    {
        "question": "Кто открыл теорию относительности?",
        "context": """
        Теория относительности была открыта Альбертом Эйнштейном и является одной из важнейших теорий в современной физике.
        """,
        "answer": "Альберт Эйнштейн"
    },
    {
        "question": "Что такое биология?",
        "context": """
        Биология — это наука, изучающая жизнь и живые организмы, включая их строение, функции, развитие и эволюцию.
        """,
        "answer": "наука"
    },
    {
        "question": "Кто написал 'Евгения Онегина'?",
        "context": """
        Роман в стихах 'Евгений Онегин' был написан русским поэтом Александром Сергеевичем Пушкиным и является одним из ключевых произведений русской классической литературы.
        """,
        "answer": "Александр Сергеевич Пушкин"
    },
    {
        "question": "Что такое блокчейн?",
        "context": """
        Блокчейн — это распределённая база данных, которая используется для записи транзакций в цифровых валютах, таких как биткойн. Блокчейн обеспечивает прозрачность и безопасность данных.
        """,
        "answer": "распределённая база данных"
    },
    {
        "question": "Что такое HTML?",
        "context": """
        HTML (HyperText Markup Language) — это язык разметки для создания веб-страниц. Он используется для структурирования контента на веб-сайтах, таких как текст, изображения и ссылки.
        """,
        "answer": "язык разметки"
    },
    {
        "question": "Кто был первым человеком на Луне?",
        "context": """
        Первым человеком, ступившим на поверхность Луны, был американский астронавт Нил Армстронг в ходе миссии Apollo 11 в 1969 году.
        """,
        "answer": "Нил Армстронг"
    },
    {
        "question": "Что такое гравитация?",
        "context": """
        Гравитация — это сила притяжения, действующая между всеми объектами, обладающими массой. Гравитация удерживает планеты на орбитах вокруг Солнца и вызывает падение объектов на Землю.
        """,
        "answer": "сила притяжения"
    },
    {
        "question": "Кто написал 'Анну Каренину'?",
        "context": """
        Роман 'Анна Каренина' был написан Львом Николаевичем Толстым и опубликован в 1877 году. Он является классикой русской литературы.
        """,
        "answer": "Лев Николаевич Толстой"
    },
    {
        "question": "Что такое интернет?",
        "context": """
        Интернет — это глобальная сеть, соединяющая миллионы компьютеров и других устройств, которая позволяет передавать информацию и получать доступ к ресурсам по всему миру.
        """,
        "answer": "глобальная сеть"
    },
    {
        "question": "Что такое черная дыра?",
        "context": """
        Черная дыра — это область пространства с чрезвычайно сильной гравитацией, которая не позволяет даже свету покинуть её пределы. Черные дыры образуются при коллапсе массивных звезд.
        """,
        "answer": "область пространства с сильной гравитацией"
    },
    {
        "question": "Что такое SQL?",
        "context": """
        SQL (Structured Query Language) — это язык запросов, который используется для управления данными в реляционных базах данных. С помощью SQL можно создавать, изменять и удалять данные.
        """,
        "answer": "язык запросов"
    },
    {
        "question": "Кто написал 'Братьев Карамазовых'?",
        "context": """
        Роман 'Братья Карамазовы' был написан русским писателем Фёдором Михайловичем Достоевским и опубликован в 1880 году.
        """,
        "answer": "Фёдор Михайлович Достоевский"
    },
    {
        "question": "Что такое нейроны?",
        "context": """
        Нейроны — это клетки нервной системы, которые передают электрические и химические сигналы между различными частями тела и мозгом. Они являются основными строительными блоками нервной системы.
        """,
        "answer": "клетки нервной системы"
    },
    {
        "question": "Что такое климат?",
        "context": """
        Климат — это долгосрочные погодные условия, характерные для определённого региона. Климат включает такие параметры, как температура, осадки и влажность.
        """,
        "answer": "долгосрочные погодные условия"
    },
    {
        "question": "Что такое энергия?",
        "context": """
        Энергия — это способность выполнять работу или производить изменения. Энергия существует в различных формах, таких как тепловая, электрическая, химическая и механическая.
        """,
        "answer": "способность выполнять работу"
    },
    {
        "question": "Что такое демография?",
        "context": """
        Демография — это наука, изучающая численность, состав и динамику населения, а также факторы, влияющие на его изменения.
        """,
        "answer": "наука, изучающая численность и состав населения"
    },
    {
        "question": "Кто написал 'Идиота'?",
        "context": """
        Роман 'Идиот' был написан Фёдором Михайловичем Достоевским в 1869 году и является классическим произведением русской литературы.
        """,
        "answer": "Фёдор Михайлович Достоевский"
    },
    {
        "question": "Что такое скорость света?",
        "context": """
        Скорость света в вакууме составляет примерно 299 792 458 метров в секунду. Это фундаментальная физическая константа, которая важна для многих разделов физики.
        """,
        "answer": "299 792 458 метров в секунду"
    },
    {
        "question": "Кто написал роман 'Герой нашего времени'?",
        "context": """
        Роман 'Герой нашего времени' был написан русским писателем Михаилом Юрьевичем Лермонтовым и является одним из шедевров русской литературы.
        """,
        "answer": "Михаил Юрьевич Лермонтов"
    },
    {
        "question": "Что такое атмосфера Земли?",
        "context": """
        Атмосфера Земли — это слой газов, который окружает планету и защищает её от космической радиации. Атмосфера состоит из различных газов, таких как азот и кислород.
        """,
        "answer": "слой газов"
    },
    {
        "question": "Что такое фотосинтез?",
        "context": """
        Фотосинтез — это процесс, при котором растения и некоторые микроорганизмы используют энергию света для синтеза органических веществ из углекислого газа и воды.
        """,
        "answer": "процесс, при котором растения используют энергию света"
    },
    {
        "question": "Кто написал 'Капитанскую дочку'?",
        "context": """
        Повесть 'Капитанская дочка' была написана Александром Сергеевичем Пушкиным и описывает события восстания Пугачёва.
        """,
        "answer": "Александр Сергеевич Пушкин"
    },
    {
        "question": "Что такое глобализация?",
        "context": """
        Глобализация — это процесс интеграции стран и народов мира, связанный с обменом товарами, услугами, информацией и культурой через границы.
        """,
        "answer": "процесс интеграции стран и народов"
    },
    {
        "question": "Кто написал 'Маленького принца'?",
        "context": """
        Книга 'Маленький принц' была написана французским писателем Антуаном де Сент-Экзюпери и является одной из самых известных книг в мировой литературе.
        """,
        "answer": "Антуан де Сент-Экзюпери"
    },
    {
        "question": "Что такое эволюция?",
        "context": """
        Эволюция — это процесс изменения живых организмов с течением времени, который приводит к появлению новых видов и адаптаций.
        """,
        "answer": "процесс изменения живых организмов"
    },
    {
        "question": "Что такое симметрия?",
        "context": """
        Симметрия — это свойство объекта оставаться неизменным при определённых преобразованиях, таких как вращение, отражение или перемещение.
        """,
        "answer": "свойство объекта оставаться неизменным"
    },
    {
        "question": "Кто написал 'Алису в стране чудес'?",
        "context": """
        'Алиса в стране чудес' — это книга, написанная английским писателем Льюисом Кэрроллом и впервые опубликованная в 1865 году.
        """,
        "answer": "Льюис Кэрролл"
    },
    {
        "question": "Что такое радиация?",
        "context": """
        Радиация — это процесс испускания энергии в виде частиц или электромагнитных волн. Она может быть природной или искусственной и используется в медицине и промышленности.
        """,
        "answer": "процесс испускания энергии"
    },
    {
        "question": "Что такое электромагнитное излучение?",
        "context": """
        Электромагнитное излучение — это форма энергии, которая распространяется в виде волн, таких как свет, радиоизлучение и рентгеновские лучи.
        """,
        "answer": "форма энергии"
    },
    {
        "question": "Что такое атом?",
        "context": """
        Атом — это основная структурная единица материи, состоящая из ядра, содержащего протоны и нейтроны, и окружающих его электронов.
        """,
        "answer": "основная структурная единица материи"
    },
    {
        "question": "Кто написал 'Лолиту'?",
        "context": """
        Роман 'Лолита' был написан Владимиром Набоковым и стал одним из самых обсуждаемых произведений 20-го века.
        """,
        "answer": "Владимир Набоков"
    },
    {
        "question": "Что такое молекула?",
        "context": """
        Молекула — это группа двух или более атомов, связанных между собой химическими связями. Молекулы являются основными строительными блоками веществ.
        """,
        "answer": "группа двух или более атомов"
    },
    {
        "question": "Что такое теорема Пифагора?",
        "context": """
        Теорема Пифагора утверждает, что в прямоугольном треугольнике квадрат гипотенузы равен сумме квадратов двух катетов.
        """,
        "answer": "квадрат гипотенузы равен сумме квадратов катетов"
    },
    {
        "question": "Что такое сингулярность?",
        "context": """
        Сингулярность — это состояние, в котором гравитационное поле бесконечно велико, и классические законы физики перестают действовать, например, внутри черной дыры.
        """,
        "answer": "состояние, в котором гравитационное поле бесконечно велико"
    },
    {
        "question": "Что такое антивещество?",
        "context": """
        Антивещество — это форма материи, состоящая из античастиц, таких как позитроны и антинейтроны, которые имеют противоположные свойства по сравнению с обычными частицами.
        """,
        "answer": "форма материи"
    },
    {
        "question": "Кто написал 'Отцы и дети'?",
        "context": """
        Роман 'Отцы и дети' был написан Иваном Сергеевичем Тургеневым и является одним из важнейших произведений русской литературы XIX века.
        """,
        "answer": "Иван Сергеевич Тургенев"
    },
    {
        "question": "Что такое плазма?",
        "context": """
        Плазма — это четвёртое состояние вещества, в котором атомы и молекулы ионизированы, и оно встречается в звездах и молниях.
        """,
        "answer": "четвёртое состояние вещества"
    },
    {
        "question": "Кто открыл периодический закон?",
        "context": """
        Периодический закон был открыт Дмитрием Ивановичем Менделеевым, который создал периодическую таблицу элементов, упорядочив их по атомным номерам.
        """,
        "answer": "Дмитрий Иванович Менделеев"
    },
    {
        "question": "Что такое термодинамика?",
        "context": """
        Термодинамика — это раздел физики, изучающий процессы, связанные с теплотой, энергией и работой, и законы, которые описывают эти процессы.
        """,
        "answer": "раздел физики"
    }
]


def answer_question(question, context, tokenizer, model):
    inputs = tokenizer.encode_plus(question, context, return_tensors='pt', max_length=512, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)


    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1  #+1, чтобы включить последний токен

    input_ids = inputs['input_ids'][0][answer_start:answer_end]
    answer = tokenizer.decode(input_ids, skip_special_tokens=True)

    return answer.strip()

def compute_accuracy(pred_answer, true_answer):
    return 1 if pred_answer.lower() == true_answer.lower() else 0

results = []

for checkpoint in checkpoints:
    checkpoint_dir = os.path.join(checkpoints_dir, checkpoint)

    tokenizer = RobertaTokenizerFast.from_pretrained(checkpoint_dir)
    model = RobertaForQuestionAnswering.from_pretrained(checkpoint_dir)
    model.to(device)

    print(f"Проверка модели {checkpoint}...")

    correct = 0
    total = len(test_data)


    for item in test_data:
        question = item["question"]
        context = item["context"]
        true_answer = item["answer"]

        pred_answer = answer_question(question, context, tokenizer, model)
        correct += compute_accuracy(pred_answer, true_answer)

    accuracy = correct / total
    results.append((checkpoint, accuracy))

    with open(os.path.join(output_dir, "test_results.txt"), "a") as f:
        f.write(f"Модель: {checkpoint}, Точность: {accuracy:.2f}\n")


checkpoints_names = [r[0] for r in results]
accuracies = [r[1] for r in results]

print("Тестирование завершено. Результаты сохранены.")
