from sklearn import tree
from sklearn import metrics
from matplotlib import pyplot as plt
import pandas as pd
import graphviz

OUTPUT_PATH = "output/"
CSV_PATH = "Obesity.csv"

# available: "matplotlib", "graphviz"
TREE_VISUALIZATION = "graphviz"

MAX_DECISION_DEPTH = 5
TRAIN_FRAC = 0.7
TEST_FRAC = 0.3

_DEBUG = True


# todo: translate ukrainian texts


def msg(message, wait_response=True, double_LF=True):
    if wait_response is True and _DEBUG is False:
        end_msg = "\n" if double_LF else ""
        print(f"> {message}")
        input(" press enter ".center(20, "-") + end_msg)
    else:
        end_msg = "\n\n" if double_LF else "\n"
        print(f"> {message}", end=end_msg)


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
NEW_CATEGORIES = {}     # { col_name : {label: int}, ... }

# change text value to the numeric - we can't use strings for DecisionTreeClassifier.fit
for col in df.columns:
    if col in EXCEPT_COLS:
        # count values, make range and number all values in column
        labels = df[col].unique()
        NEW_CATEGORIES[col] = {id_: label for id_, label in enumerate(labels, 1)}
        msg(f"Created new category ids for column {col!r}: {NEW_CATEGORIES[col]}")
        for i, label in enumerate(labels, 1):
            df.loc[df[col] == label, col] = i
    else:
        for label, val in CATEGORY_REPLACE.items():
            df.loc[df[col] == label, col] = val

    if df[col].dtype == "O":
        # if dtype of column is 'object' (string) - cast to float
        df[col] = df[col].astype("float")

msg(f"Ми замінили усюди текстові категорії на числа: {CATEGORY_REPLACE}, "
    f"а для колонок {EXCEPT_COLS} ми просто пронумеровали від 1 до кількості категорій стовпця:\n{df.head(10)}",
    double_LF=False, wait_response=False)
msg(f"На всяк випадок впевнимося, що типи даних датафрейму саме числові:\n{df.dtypes}")

tr_df = df.sample(frac=TRAIN_FRAC)
msg(f"training sample (random {TRAIN_FRAC*100}% from our df):\n{tr_df}",
    wait_response=False, double_LF=False)
test_df = df.sample(frac=TEST_FRAC)
msg(f"test sample (random {TEST_FRAC*100}% from our df):\n{test_df}")

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
msg(f"Create DecisionTreeClassifier with max_depth={MAX_DECISION_DEPTH}",
    double_LF=False)
classifier = tree.DecisionTreeClassifier(max_depth=MAX_DECISION_DEPTH)
classifier.fit(tr_x, tr_y)
test_predict = classifier.predict(test_x)

test_predict_accuracy = metrics.accuracy_score(test_y, test_predict)
_accuracy_percent = round(test_predict_accuracy*100, 2)


def get_satisfy_label(percent):
    # ONLY WORKS FOR PYTHON 3 (coz in python 2 dict has no order)
    SATISFY_PERCENTS = {
        100: "excellent",
        98: "awesome",
        95: "great",
        75: "good",
        50: "random",
        0: "problematic"
    }
    for t, label in SATISFY_PERCENTS.items():
        if percent >= t:
            return label
    return "error"


msg(f"classifier Accuracy is {_accuracy_percent}%: " + get_satisfy_label(_accuracy_percent))


# 6. Представити графічно побудоване дерево за допомогою бібліотеки graphviz.


def showMatplotlibTree(decision_tree, _feature_names, _class_names):
    fig, axes = plt.subplots(figsize=(50, 15), dpi=300)
    plt.tight_layout()
    tree.plot_tree(decision_tree, filled=True, ax=axes, fontsize=10,
                   feature_names=_feature_names, class_names=_class_names)
    fig.savefig(OUTPUT_PATH + "matplotlib_tree.png")


def showGraphvizTree(decision_tree, _feature_names, _class_names):
    dot_data = tree.export_graphviz(decision_tree, filled=True, out_file="tree.dot",
                                    feature_names=_feature_names, class_names=_class_names)
    graph = graphviz.Source(dot_data, format="png")
    graph.render("graphviz_tree.png", directory=OUTPUT_PATH, overwrite_source=True)


feature_names = tr_x.columns
class_names = list(NEW_CATEGORIES['NObeyesdad'].values())

SHOW_TREE_FN = {
    "graphviz": showGraphvizTree,
    "matplotlib": showMatplotlibTree
}

if _DEBUG is True:
    for lib, show_tree in SHOW_TREE_FN.items():
        try:
            show_tree(classifier, feature_names, class_names)
        except Exception as e:
            msg(f"<!> An error occurred while show_tree using {lib!r}: {e}")
else:
    show_tree = SHOW_TREE_FN.get(TREE_VISUALIZATION, "matplotlib")
    show_tree(classifier, feature_names, class_names)

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
