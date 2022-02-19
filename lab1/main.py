import matplotlib.pyplot as plt
import pandas as pd


csv_name = "Top100-2007.csv"
k = 11


# 1. Відкрити та зчитати файл з даними.
df = pd.read_csv(csv_name)
print(f"> {csv_name!r} successfully read\n")

# 2. Визначити та вивести кількість записів та кількість полів у кожному записі.
total_rows, total_columns = df.shape
print(f"> {csv_name!r} has {total_rows} rows (записи) and {total_columns} columns (поля)\n")

# 3. Вивести 5 записів, починаючи з К-ого, та 3К+2 останніх записів,
# де число К визначається днем народження студента та має бути визначено як змінна.
five_records = df.iloc[k:k+5]
print(f"> 5 records from k={k} to k+5={k+5}:\n{five_records}\n")
last_records = df.tail(3*k+2)
print(f"> last 3k+2=35 records:\n{last_records}\n")

# 4. Визначити та вивести тип полів кожного запису.
col_types = df.dtypes
print(f"> automatic types in our table ('object' is str):\n{col_types}\n")

# 5. Очистити текстові поля від зайвих пробілів.
print(f"> country from first record BEFORE removing spaces: {df.iloc[0].Country!r}\n")
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
print(f"> country from first record AFTER removing spaces: {df.iloc[0].Country!r}\n")

# 6. Визначити поля, які потрібно привести до числового вигляду та зробити це (продемонструвати підтвердження).
cast_details = {
    "Winning Percentage": {
        "type": "float",
        "remove": "%"
    },
    "Career Earnings": {
        "type": "int",
        "remove": "$"
    },
}
print("> We should change types for next columns:")
for col, details in cast_details.items():
    cast_type = details["type"]
    remove_symbol = details["remove"]
    print(f">  {col!r} from '{df[col].dtype}' to {cast_type!r}", end="")
    df[col] = df[col].apply(lambda x: x.replace(remove_symbol, "") if isinstance(x, str) else x)
    df[col] = df[col].astype(cast_type, errors="ignore")
    print(f" -> result type: {df[col].dtype}")
print("\n> As we can see, we couldn't cast 'Career Earnings' to int because of NaN values...\n")

# 7. Визначити записи із пропущеними даними та вивести їх на екран, після чого видалити з датафрейму.
print("> Records with NaN values:")
print(df[df.isnull().any(1)], end="\n\n")
df = df.dropna()
print("> We delete them. Check it one more times:")
print(df[df.isnull().any(1)], end="\n\n")

# 8. На основі поля Singles Record (Career) ввести нові поля (формат: победы-поражения)
# code...

# 9. Видалити з датафрейму поля Singles Record (Career) та Link to Wikipedia.
drop_list = ["Singles Record (Career)", "Link to Wikipedia"]
df = df.drop(drop_list, axis=1)
print(df)

# 10. Змінити порядок розташування полів
# new_order = ["Rank", "Name", "Country", "Pts", "Total", "Win", "Lose", "Winning Percentage", "Career Earnings"]
__last_order = df.columns.values
# df = df.loc[:, new_order]
print(f"> Changed column's order from {__last_order} to {df.columns.values}")

