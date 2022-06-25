from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.model_selection import ShuffleSplit
from sklearn import linear_model
import pandas as pd

OUTPUT_PATH = "output/"
CSV_PATH = "Obesity.csv"
TRAIN_FRAC = 0.7
TEST_FRAC = 0.3

_DEBUG = True                      # Do not wait response from method msg


def msg(message, wait_response=True, double_LF=True):
    if wait_response is True and _DEBUG is False:
        end_msg = "\n" if double_LF else ""
        print(f"> {message}")
        input(" "*4 + " press enter ".center(20, "-") + end_msg)
    else:
        end_msg = "\n\n" if double_LF else "\n"
        print(f"> {message}", end=end_msg)


# 1. Відкрити та зчитати наданий файл з даними.
df = pd.read_csv(CSV_PATH)
df.rename(columns={"family_history_with_overweight": "family_overweight"}, inplace=True)
msg(f"1. Successfully read csv {CSV_PATH}",
    wait_response=False)

# 2. Визначити та вивести кількість записів та кількість полів у кожному записі.
total_rows, total_columns = df.shape
msg(f"2. {CSV_PATH!r} has {total_rows} rows (записи) and {total_columns} columns (поля). Типи даних:",
    wait_response=False)
msg(df.dtypes)

# 3. Вивести перші 10 записів набору даних.
msg(f"3. First 10 records from csv:\n{df.head(10)}")

# 4. Додати до набору даних атрибут NormalWeight,
# який дорівнює 1, якщо NObeyesdad приймає значення Normal_Weight,
# та 0 в інших випадках.
df["NormalWeight"] = 0
df.loc[df["NObeyesdad"] == "Normal_Weight", "NormalWeight"] = 1

# 5. Вивести перші 10 записів набору даних
# та видалити після цього атрибут NObeyesdad з набору даних.
msg(f"4. Add new column 'NormalWeight', where 1 means 'Normal_Weight' and 0 other values...",
    wait_response=False, double_LF=False)
msg(f"5. First 10 records:\n{df.head(10)}", double_LF=False)
msg(f"Delete column 'NObeyesdad'...")
df.pop("NObeyesdad")


# 6. Перемішати набір даних та розділити його на навчальну (тренувальну)
# та тестову вибірки, використовуючи функцію ShuffleSplit.
msg(f"6. Оскільки в нашому датасеті присутній текст (типу категорії), треба перевести його в числа!")
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
EXCEPT_COLS = ["MTRANS"]
NEW_CATEGORIES = {}     # { col_name : {label: int}, ... }

# change text value to the numeric - we can't use strings for DecisionTreeClassifier.fit
for col in df.columns:
    if col in EXCEPT_COLS:
        # count values, make range and number all values in column
        labels = df[col].unique()
        NEW_CATEGORIES[col] = {id_: label for id_, label in enumerate(labels, 0)}
        msg(f"Created new category ids for column {col!r}: {NEW_CATEGORIES[col]}",
            double_LF=False, wait_response=False)
        for i, label in enumerate(labels, 0):
            df.loc[df[col] == label, col] = i
    else:
        for label, val in CATEGORY_REPLACE.items():
            df.loc[df[col] == label, col] = val

    if df[col].dtype == "O":
        # if dtype of column is 'object' (string) - cast to float
        df[col] = df[col].astype("float")

msg(f"Ми замінили усюди текстові категорії на числа: {CATEGORY_REPLACE}, "
    f"а для колонок {EXCEPT_COLS} ми просто пронумеровали від 0 до кількості категорій стовпця:\n{df.head(10)}")

msg("Розділимо вибірки на тренувальну та тестову за допомогою ShuffleSplit")
rs = ShuffleSplit(n_splits=1, test_size=TEST_FRAC, train_size=TRAIN_FRAC, random_state=0)

X, y = df.iloc[:, :-1], df.iloc[:, -1]
tr_index, test_index = list(rs.split(X, y))[0]

tr_x, tr_y = X.iloc[tr_index], y.iloc[tr_index]
msg(f"training sample (ShuffleSplit {TRAIN_FRAC*100}% from our df):"
    f"\n{pd.concat([tr_x, tr_y], axis=1)}", double_LF=False)
test_x, test_y = X.iloc[test_index], y.iloc[test_index]
msg(f"test sample (ShuffleSplit {TEST_FRAC*100}% from our df):"
    f"\n{pd.concat([test_x, test_y], axis=1)}")

# 7. Використовуючи відповідні функції бібліотеки scikit-learn,
# збудувати класифікаційну модель логістичної регресії та навчити її
# натренувальній вибірці, вважаючи, що цільова характеристика
# визначається стовпчиком NormalWeight, а всі інші виступають в ролі вихідних аргументів.
msg(f"7. Create LogisticRegression")
logistic = linear_model.LogisticRegression(random_state=0, max_iter=500)
logistic.fit(tr_x, tr_y)


# 8. Обчислити класифікаційні метрики збудованої моделі для тренувальної та тестової вибірки.
def report_metrics(X, y, classifier, sample_label="", plt_show=True):
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

    print("\n", f" metrics of {sample_label} sample ".center(50, "-"))

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
    print(f" - F1 score is {_F1_percent}% (Оцінка класифікатора - середнє гармонічне precision та recall)")

    print("-"*50, "\n")

    if plt_show is True:
        plt.show(block=True)

    return pred


msg("Get metrics from prediction of test sample", double_LF=False)
test_predict = report_metrics(test_x, test_y, logistic, "test", plt_show=True)
msg("Get metrics from prediction of training sample", double_LF=False)
tr_predict = report_metrics(tr_x, tr_y, logistic, "train", plt_show=True)

# 9. З’ясувати вплив точності для критерію зупинки ітерацій
# (від 0.0001 до 0.1) на результати класифікації. Результати представити графічно.
msg("9. З'ясуємо вплив точності критерія зупинки ітерацій (від 0.0001 до 0.1) на точність класифікатора")
accs = []
for i in range(1, 5):
    tol = 10**-i
    _logistic = linear_model.LogisticRegression(tol=tol, max_iter=500, random_state=0)
    _logistic.fit(tr_x, tr_y)
    pred = _logistic.predict(test_x)
    acc = round(metrics.accuracy_score(test_y, pred)*100, 1)
    accs.append(acc)

fig, ax = plt.subplots()
ax.bar(range(len(accs)), accs)
ax.set_xlabel("Точність критерія зупинки")
ax.set_ylabel("Точність класифікатора, %")
ax.set_title("Вплив точності критерія зупинки ітерацій на точність класифікатора")
ax.set_xticks(range(len(accs)))
ax.set_xticklabels([f"1e-{i+1}" for i in range(len(accs))])
for i, acc in enumerate(accs):
    ax.annotate(f"{acc}%", (i-0.15, acc/2))
fig.savefig(OUTPUT_PATH+"task9.png")
plt.show(block=True)

msg("Отже, точність критерія зупинки не вплинула на точність класифікатора")

# 10. Проаналізувати ступінь впливу атрибутів на результат класифікації.
# Збудувати класифікаційну модель логістичної регресії,
# залишивши від 3 до 5 найбільш важливих атрибутів та порівняти її результати із моделлю з п. 7.
msg("10. Show feature importances of classification model (opens matplotlib graphic)")


def show_feature_importances(lm):
    fig, axes = plt.subplots()
    feature_names = tr_x.columns
    axes.bar(feature_names, lm.coef_[0])
    axes.set_title("feature importances of classification model")
    plt.xticks(rotation=90)
    plt.tight_layout()
    fig.savefig(OUTPUT_PATH + "feature_importances_bar.png")
    plt.show(block=True)


show_feature_importances(logistic)
importances = ["Gender", "Height", "FAVC", "CAEC", "SMOKE"]
msg(f"Як бачимо з графіка, параметри {importances} впливають більше всього. Їх і залишимо!")

tr_x2 = tr_x[importances]
test_x2 = test_x[importances]
logistic2 = linear_model.LogisticRegression(max_iter=500, random_state=0)
logistic2.fit(tr_x2, tr_y)
msg("Get metrics from prediction of truncated test sample", double_LF=False)
report_metrics(test_x2, test_y, logistic2, "truncated-test")
msg("В результаті: точність збільшилась на 2 відсотки")

input(" press Enter to finish ".center(20, "="))
