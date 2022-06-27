from sklearn import cluster
from sklearn import neighbors
import pandas as pd
import numpy as np


CSV_PATH = "Obesity.csv"
TRAIN_FRAC = 0.7
TEST_FRAC = 0.3

_DEBUG = False   # Do not wait response from method msg


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
msg(f"2. {CSV_PATH!r} has {total_rows} rows (записи) and {total_columns} columns (поля).",
    wait_response=False)


# підготуємо дані...
def prepare(_df):
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
    EXCEPT_COLS = ["MTRANS", "NObeyesdad"]
    NEW_CATEGORIES = {}     # { col_name : {label: int}, ... }

    # change text value to the numeric - we can't use strings for DecisionTreeClassifier.fit
    for col in _df.columns:
        if col in EXCEPT_COLS:
            # count values, make range and number all values in column
            labels = _df[col].unique()
            NEW_CATEGORIES[col] = {id_: label for id_, label in enumerate(labels, 0)}
            # msg(f"Created new category ids for column {col!r}: {NEW_CATEGORIES[col]}",
            #     double_LF=False, wait_response=False)
            for i, label in enumerate(labels, 0):
                _df.loc[_df[col] == label, col] = i
        else:
            for label, val in CATEGORY_REPLACE.items():
                _df.loc[_df[col] == label, col] = val

        if _df[col].dtype == "O":
            # if dtype of column is 'object' (string) - cast to float
            _df[col] = _df[col].astype("float")

    msg(f"Ми замінили усюди текстові категорії на числа: {CATEGORY_REPLACE},\n"
        f"а для колонок {EXCEPT_COLS} ми просто пронумеровали від 0 до кількості категорій стовпця:\n{_df.head(5)}")

    return _df


categories = df["NObeyesdad"].unique()
categories = {i: cat for i, cat in enumerate(categories)}
df = prepare(df)


# 3. Видалити атрибут NObeyesdad
msg("3. Delete attribute 'NObeyesdad'...")
y = df.pop("NObeyesdad")


# 4. Вивести атрибути, що залишилися.
msg(f"4. Attributes:\n{list(df.columns)}")


# 5. Використовуючи функцію KMeans бібліотеки scikit-learn,
# виконати розбиття набору даних на кластери. Кількість кластерів
# визначити на основі початкового набору даних (вибір обгрунтувати).
msg(f"5. KMeans: Розіб'ємо дані на {len(categories)} кластерів, оскільки маємо саме {len(categories)} варіантів цільвого значення: {categories}.")
kmeans = cluster.KMeans(n_clusters=len(categories))
kmeans.fit(df)
# kmeans_df = pd.concat([df, y, pd.Series(kmeans.labels_, name="cluster")], axis=1)
# kmeans_df.groupby("NObeyesdad")["cluster"].value_counts()


# Вивести координати центрів кластерів.
def enum_centers(centers):
    msg("Центри кластерів:", wait_response=False)
    for i, center in enumerate(centers):
        print(f"[{i}] {', '.join([f'{c:.4f}' for c in center])}")
    msg("")


enum_centers(kmeans.cluster_centers_)


# 6. Використовуючи функцію AgglomerativeClustering бібліотеки scikit-learn,
# виконати розбиття набору даних на кластери. Кількість кластерів обрати
# такою ж самою, як і в попередньому методі. Вивести координати центрів кластерів.
msg(f"6. AgglomerativeClustering: Розіб'ємо дані на також {len(categories)} кластерів.")
aggl = cluster.AgglomerativeClustering(n_clusters=len(categories))
aggl_pred = aggl.fit_predict(df)
near_aggl = neighbors.NearestCentroid().fit(df, aggl_pred)
enum_centers(near_aggl.centroids_)


# 7. Порівняти результати двох використаних методів кластеризації.
msg("7. Порівняємо, наскільки центри обох методів відрізняютья. ")


def calc_dif(centers1, centers2):
    dif_centers = []
    dif_means = []
    for i, (c1, c2) in enumerate(zip(centers1, centers2)):
        dif = np.abs(c1-c2)
        mean = np.mean(dif)
        dif_centers.append(dif)
        dif_means.append(mean)

        print(f"[{i}] В середньому відрізняються: {mean:.4f}"
              f" | Більше трьох середніх: {len(list(filter(lambda x: x > 3*mean, dif)))}"
              f" | Абсолютні різниці: {', '.join([f'{c:.4f}' for c in dif])}")
    msg(f"\nВ середньому, усі різниці відхиляються на {np.mean(dif_means):.4f} у.о.")
    return dif_centers


calc_dif(kmeans.cluster_centers_, near_aggl.centroids_)
msg(f"Висновок: обидва метода дали +\\- однакові результати, "
    f"але, як правило, спостерігаються 1-2 розбіжності для певної характеристики")


input(" press Enter to finish ".center(20, "="))
