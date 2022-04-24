import pandas as pd
from sklearn.tree import DecisionTreeClassifier


# todo: translate ukrainian texts


def msg(message, wait_response=True, double_LF=True):
    if wait_response is True:
        end_msg = "\n" if double_LF else ""
        print(f"> {message}")
        input(" press enter ".center(20, "-"))
    else:
        end_msg = "\n\n" if double_LF else "\n"
        print(f"> {message}", end=end_msg)


CSV_PATH = "Obesity.csv"


# 1. Відкрити та зчитати наданий файл з даними.
df = pd.read_csv(CSV_PATH)
msg(f"1. successfully read csv {CSV_PATH}",
    wait_response=False)

# 2. Визначити та вивести кількість записів та кількість полів у кожному записі.
total_rows, total_columns = df.shape
msg(f"2. {CSV_PATH!r} has {total_rows} rows (записи) and {total_columns} columns (поля)",
    wait_response=False)

# 3. Вивести перші 10 записів набору даних.
msg(f"3. first 10 records from csv:\n{df.head(10)}")

# 4. Розділити набір даних на навчальну (тренувальну) та тестову вибірки.
msg(f"4. Оскільки в нашому датасеті присутній текст (типу категорії), треба перевести його в числа!")
CATEGORY_REPLACE = {
    # if we found key in df column - replace to value according this key
    "yes": 1,
    "no": 0,
    "Sometimes": 0.33,
    "Frequently": 0.66,
    "Always": 1,
    "Female": 1,
    "Male": 0
}
EXCEPT_COLS = ["MTRANS", "NObeyesdad", ]

for col in df.columns:
    if col in EXCEPT_COLS:
        # count values, make range and number all values in column
        labels = df[col].unique()
        for i, label in enumerate(labels, 1):
            df.loc[df[col] == label, col] = i
    else:
        for label, val in CATEGORY_REPLACE.items():
            df.loc[df[col] == label, col] = val
    if df[col].dtype == "O":
        df[col] = df[col].astype("float")

msg(f"Ми замінили усюди текстові категорії на числа: {CATEGORY_REPLACE}, "
    f"а для колонок {EXCEPT_COLS} ми просто пронумеровали від 1 до кількості категорій стовпця:\n{df.head(10)}",
    double_LF=False, wait_response=False)
msg(f"На всяк випадок впевнимося, що типи даних датафрейму саме числові:\n{df.dtypes}")

tr_df = df.sample(frac=0.7)
msg(f"training sample (random 70% from our df):\n{tr_df.head()}",
    wait_response=False, double_LF=False)
test_df = df.sample(frac=0.3)
msg(f"test sample (random 30% from our df):\n{test_df}")

# 5. Використовуючи scikit-learn збудувати дерево прийняття рішен глибини 5
#    та навчити її на тренувальній вибірці, вважаючи, що в наданому наборі даних
#    цільова характеристика визначається останнім стовпчиком, а всі інші
#    виступають в ролі вихідних аргументів.

# Поділили вибірки на вихідні аргументі та цільову характеристику (стовпець NObeyesdad)
tr_x, tr_y = tr_df.iloc[:, :-1], tr_df.iloc[:, -1]
test_x, test_y = test_df.iloc[:, :-1], test_df.iloc[:, -1]
msg(f"5. Поділили вибірки на вихідні аргументи та цільову характеристику (це останній стовпець 'NObeyesdad')",
    wait_response=False, double_LF=False)

# Навчимося на тренувальних даних
classifier = DecisionTreeClassifier(max_depth=5)
classifier.fit(tr_x, tr_y)
# todo: глянуть, что можно принтануть для пруфа

# todo: 6. Представити графічно побудоване дерево за допомогою бібліотеки graphviz.
pass

# todo: 7.1. Обчислити класифікаційні метрики збудованої моделі для тренувальної та тестової вибірки.
pass

# todo 7.2. Представити результати роботи моделі на тестовій вибірці графічно.
pass

# todo 7.3. Порівняти результати, отриманні при застосуванні різних критеріїв розщеплення:
#  інформаційний приріст на основі ентропії чи неоднорідності Джині.
pass

# todo 8. З’ясувати вплив максимальної кількості листів та мінімальної кількості
#  елементів в листі дерева на результати класифікації. Результати представити графічно.
pass

# todo 9. Навести стовпчикову діаграму важливості атрибутів, які використовувалися для класифікації
#  (див. feature_importances_). Пояснити, яким чином – на Вашу думку – цю важливість можна підрахувати
pass


input(" press Enter to finish ".center(20, "="))
