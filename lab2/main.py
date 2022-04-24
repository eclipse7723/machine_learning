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

_DEBUG = True                       # Do not wait response from method msg
_DEBUG_CREATE_GRAPH = True          # If False - no tree graphs would be created
_DEBUG_SELECTED_GRAPHS_LIB = True   # If True use graph lib from TREE_VISUALIZATION else use all available


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
gini_classifier = tree.DecisionTreeClassifier(max_depth=MAX_DECISION_DEPTH, criterion='gini')
gini_classifier.fit(tr_x, tr_y)

# 6. Представити графічно побудоване дерево за допомогою бібліотеки graphviz.


def _showMatplotlibTree(decision_tree, _feature_names, _class_names, _suffix=""):
    msg(f"Create tree graph to {OUTPUT_PATH}matplotlib_decision_tree{_suffix}.png using lib matplotlib",
        double_LF=False, wait_response=False)
    fig, axes = plt.subplots(figsize=(50, 15), dpi=300)
    plt.tight_layout()
    tree.plot_tree(decision_tree, filled=True, ax=axes, fontsize=10,
                   feature_names=_feature_names, class_names=_class_names)
    fig.savefig(OUTPUT_PATH + f"matplotlib_tree{_suffix}.png")


def _showGraphvizTree(decision_tree, _feature_names, _class_names, _suffix=""):
    msg(f"Create tree graph to {OUTPUT_PATH}graphviz_decision_tree{_suffix}.png using lib graphviz",
        double_LF=False, wait_response=False)
    dot_data = tree.export_graphviz(decision_tree, filled=True,
                                    feature_names=_feature_names, class_names=_class_names)
    graph = graphviz.Source(dot_data)
    try:
        graph.render(f"graphviz_tree{_suffix}", directory=OUTPUT_PATH,
                     format="png", overwrite_source=True, cleanup=True)
    except graphviz.backend.ExecutableNotFound as RuntimeError:
        msg("<!> Failed to render graph, you need to install: https://graphviz.org/download/ and restart PC")
        raise RuntimeError


def showTree(decision_tree, _feature_names, _class_names, _suffix="", _force=False):
    SHOW_TREE_FN = {
        "graphviz": _showGraphvizTree,
        "matplotlib": _showMatplotlibTree
    }
    if all([_DEBUG is True, _DEBUG_CREATE_GRAPH is False, _force is False]):
        msg("skip graph creation _DEBUG_CREATE_GRAPH==False...")
    elif _DEBUG is True and _DEBUG_SELECTED_GRAPHS_LIB is False:
        for lib, show_tree in SHOW_TREE_FN.items():
            try:
                show_tree(decision_tree, _feature_names, _class_names, _suffix=_suffix)
            except Exception as e:
                msg(f"<!> An error occurred while show_tree using {lib!r}: {e}")
    else:
        show_tree = SHOW_TREE_FN.get(TREE_VISUALIZATION, "graphviz")
        show_tree(decision_tree, _feature_names, _class_names, _suffix=_suffix)


msg("6. Show DecisionTreeClassifier graph")
feature_names = tr_x.columns
class_names = list(NEW_CATEGORIES['NObeyesdad'].values())
showTree(gini_classifier, feature_names, class_names, "_gini")


# 7.1. Обчислити класифікаційні метрики збудованої моделі для тренувальної та тестової вибірки.
def report_metrics(X, y, classifier, sample_label=""):
    """ :return: predicted array """
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

    def in_percent(val):
        return round(val*100, 2)

    pred = classifier.predict(X)

    print(f" metrics of {sample_label} sample ".center(50, "-"))

    conf_matrix = metrics.confusion_matrix(y, pred)
    ConfusionMatrixDisplay = metrics.ConfusionMatrixDisplay(conf_matrix)
    fig, axes = plt.subplots()
    axes.set_title(f"Confusion Matrix of {sample_label} sample")
    ConfusionMatrixDisplay.plot(ax=axes)
    fig.savefig(OUTPUT_PATH+sample_label+"_confusion_matrix.png")
    print(f" You can found Confusion matrix in {OUTPUT_PATH+sample_label+'_confusion_matrix.png'}")

    accuracy = metrics.accuracy_score(y, pred)
    _accuracy_percent = in_percent(accuracy)
    print(f" - Accuracy is {_accuracy_percent}% (Доля правильних відповідей): " + get_satisfy_label(_accuracy_percent))

    accuracy = metrics.balanced_accuracy_score(y, pred)
    _accuracy_percent = in_percent(accuracy)
    print(f" - Balanced Accuracy is {_accuracy_percent}% (Збалансована влучність): " + get_satisfy_label(_accuracy_percent))

    precision = metrics.precision_score(y, pred, average='macro')
    _precision_percent = in_percent(precision)
    print(f" - Precision is {_precision_percent}% (Наскільки влучно дані класифікуються)")

    recall = metrics.recall_score(y, pred, average='macro')
    _recall_percent = in_percent(recall)
    print(f" - Recall is {_recall_percent}% (Наскільки добре класи ідентифікуються в цілому)")

    F1 = metrics.f1_score(y, pred, average='macro')
    _F1_percent = in_percent(F1)
    print(f" - F1 is {_F1_percent}% (Оцінка класифікатора - середнє гармонічне precision та recall)")

    print("-"*50)
    return pred


test_predict = report_metrics(test_x, test_y, gini_classifier, "gini_test")
tr_predict = report_metrics(tr_x, tr_y, gini_classifier, "gini_training")

# todo 7.2. Представити результати роботи моделі на тестовій вибірці графічно.
pass    # ROC curve multi class...

# todo 7.3. Порівняти результати, отриманні при застосуванні різних критеріїв розщеплення:
#  інформаційний приріст на основі ентропії чи неоднорідності Джині.
_classifier = tree.DecisionTreeClassifier(max_depth=MAX_DECISION_DEPTH, criterion='entropy')
_classifier.fit(tr_x, tr_y)
showTree(_classifier, feature_names, class_names, _suffix="_entropy", _force=True)
tr_predict_entropy = report_metrics(tr_x, tr_y, _classifier, "entropy_training")
msg("You can compare metrics of training sample for gini and entropy criterias")

# todo 8. З’ясувати вплив максимальної кількості листів та мінімальної кількості
#  елементів в листі дерева на результати класифікації. Результати представити графічно.
pass

# todo 9. Навести стовпчикову діаграму важливості атрибутів, які використовувалися для класифікації
#  (див. feature_importances_). Пояснити, яким чином – на Вашу думку – цю важливість можна підрахувати
pass


input(" press Enter to finish ".center(20, "="))
