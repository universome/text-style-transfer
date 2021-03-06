import random


def retrieve_random_dialog():
    news = random.choice(NEWS)
    dialog = convert_news_to_dialog(news)

    return dialog


def convert_news_to_dialog(news):
    paragraphs = [news['title']] + news['text'].splitlines()
    sentences = [(s + '.') for p in paragraphs for s in p.split('.')]
    lines = [{'speaker': random.choice(['Borgy', 'Bes']), 'text': s} for s in sentences]

    return lines


# Ok, let's hardcode some news currently
NEWS = [
    {
        "title": "Мы всех обыграли в хоккей! Ура!",
        "text": "Тут особого текста я придумывать не стал. Просто текст как текст."
    },
    {
        "title": "Гугл знает о детях больше, чем они сами. Я вижу в этом опасность.",
        "text": "В ноябре 2018 года американский художник и исследователь Тревор Паглен вместе с компанией SpaceX собирается вывести на орбиту «Орбитальный отражатель» — первый спутник, который не будет иметь ни военной, ни научной, ни технической ценности, а будет исключительно объектом искусства. При помощи этой акции художник хочет показать, что монополия отдельных государств и правительств на космические запуски вот-вот закончится. Незадолго до запуска спутника Паглен принял участие в Рижской биеннале современного искусства с циклом работ, посвященных подводной инфраструктуре интернета и массовой слежке за пользователями. Спецкор «Медузы» Константин Бенюмов встретился с Пагленом, чтобы обсудить, зачем художнику нужна математика и умение нырять с аквалангом, чем опасен технологический прогресс и как находить средства на создание современного искусства, не испортив репутации."
    },
    {
        "title": "Площадь льдов в Арктике стремительно сокращается.",
        "text": "Площадь льдов в Арктике стремительно сокращается. У таяния льдов есть очевидные негативные последствия, например, угроза затопления прибрежных территорий и изменение привычной среды обитания животных. Но открываются и новые перспективы — становится проще и дешевле использовать Северный морской путь.\nКомпания Maersk — мировой лидер по морским грузовым перевозкам — в конце августа отправила судно в пробный рейс по Северному морскому пути.\nВласти России намерены в ближайшие шесть лет в восемь раз увеличить объем перевозимых по этому маршруту грузов."
    },
    {
        "title": "На заводе по производству взрывчатки в Нижегородской области произошел взрыв, есть погибшие.",
        "text": "В Нижегородской области на заводе имени Свердлова, занимающегося производством взрывчатки, произошел взрыв. По данным источника «Интерфакса», погибли три человека; еще три человека пострадали.\nРИА Новости пишет, что судьба трех человек остается неизвестной — по словам источника агентства, они могут быть под завалами. После взрыва на предприятии начался пожар на площади около ста метров, сообщает агентство.\nПресс-служба завода подтвердила, что в здании для утилизации мин произошел «хлопок». «Проводится проливка территории и здания пожарной охраной. О развитии событий будем информировать дополнительно», — сказано в сообщении. О гибели людей предприятие не сообщило."
    }
]
