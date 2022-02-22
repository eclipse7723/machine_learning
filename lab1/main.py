import matplotlib.pyplot as plt
import pandas as pd
import os


# Create directory for output plots
SAVE_PICTURES = True
script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, "output")
if not os.path.isdir(results_dir) and SAVE_PICTURES is not False:
    ans = input(f"This lab creates images for tasks 13-15 in directory {results_dir},"
                f" but it doesn't exists. I'll create it, ok?\n"
                f"If no, program will creates matplotlib window with that graphs [Y/N] ")
    if ans.lower() == "y":
        os.makedirs(results_dir)
    else:
        SAVE_PICTURES = False


def save_fig(fig, title):
    if SAVE_PICTURES is False:
        plt.show(block=True)
        return
    try:
        fig.savefig(f"{results_dir}/{title}.jpg")
    except FileNotFoundError as ex:
        print(f"<!> Can't save graphic {title!r}, create folder '{results_dir}'")
        plt.show(block=True)


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
print("\n> As we can see, we couldn't cast all 'Career Earnings' rows to int because of NaN values...\n")

# 7. Визначити записи із пропущеними даними та вивести їх на екран, після чого видалити з датафрейму.
print("> Records with NaN values:")
print(df[df.isnull().any(1)], end="\n\n")
df = df.dropna()
print("> We deleted them. Check it one more times:")
print(df[df.isnull().any(1)], end="\n\n")

df['Career Earnings'] = df['Career Earnings'].astype("int")
print(f"> Also try again cast 'Career Earnings' rows to int. Current type is {df['Career Earnings'].dtype}\n")

# 8. На основі поля Singles Record (Career) ввести нові поля (формат: победы-поражения)
print(f"> 'Singles Record (Career)' is not useful for analyze, we need to fix it.")
singles_record = df.copy()["Singles Record (Career)"]
new_df = singles_record.str.split("-", expand=True).astype("int")
new_df.columns = ["Win", "Lose"]
total = []
for i, row in new_df.iterrows():
    total.append(row[0]+row[1])
new_df["Total"] = total
print(f"  > Created new dataframe with columns 'Win', 'Lose' and 'Total' from 'Singles Record (Career)':")
print(new_df)
df = df.join(new_df)
print(f"  > And add them to our main dataframe:\n{df}\n")

# 9. Видалити з датафрейму поля Singles Record (Career) та Link to Wikipedia.
drop_list = ["Singles Record (Career)", "Link to Wikipedia"]
df = df.drop(drop_list, axis=1)
print(f"> Next step is deleting columns {drop_list}. Check it:\n{df}\n")

# 10. Змінити порядок розташування полів
new_order = ["Rank", "Name", "Country", "Pts", "Total", "Win", "Lose", "Winning Percentage", "Career Earnings"]
__last_order = df.columns.values
df = df.loc[:, new_order]
print(f"> Changed column's order from {__last_order} to {df.columns.values}\n")

# 11.a. Відсортований за абеткою перелік країн, тенісисти з яких входять у Топ-100;
countries = df["Country"].sort_values().drop_duplicates()
print(f"> Top-100 sorted countries without duplicates:")
print("\n".join("   %s" % country for country in countries.values), end="\n\n")

# 11.b. Гравця та кількість його очок із найменшою сумою призових;
__df = df.loc[:, ["Name", "Pts", "Career Earnings"]].copy()
smallest_prize_player = __df[__df["Career Earnings"] == __df["Career Earnings"].min()]
smallest_prize_player = smallest_prize_player.values[0]
print(f"> Player {smallest_prize_player[0]} has the smallest"
      f" Career Earnings (${smallest_prize_player[2]}),"
      f" his Pts is {smallest_prize_player[1]}", end="\n\n")

# 11.c. Гравців та країну, яку вони представляють, кількість виграних матчів у яких дорівнює кількості програних.
__df = df[df['Win'] == df['Lose']].loc[:, ["Name", "Country"]].copy()
print(f"> These players has same number of wins and losses:\n{__df}\n")

# 12.a. Кількість тенісистів з кожної країни у Топ-100;
countries_size = pd.pivot_table(df, index=["Country"], aggfunc="size").sort_values(ascending=False)
print(f"> Number of players from each Top-100 countries:")
print(countries_size, end="\n\n")

# 12.b. Середній рейтинг тенісистів з кожної країни.
mean_rating = pd.pivot_table(df, index="Country", values="Pts", aggfunc="mean").sort_values(by="Pts", ascending=False)
print(f"> Mean pts of players from each countries:")
print(mean_rating, end="\n\n")

# 13. Побудувати діаграму кількості програних матчів по кожній десятці гравців з Топ-100.
for i in range(10):
    fig, ax = plt.subplots()
    _df = df.loc[i*10:i*10 + 10, ["Lose", "Name"]]
    _df.plot(ax=ax, kind="bar")
    ax.set_xticklabels(_df["Name"], rotation=90, ha='right')
    graph_name = f"{i*10}_{i*10 + 10}"
    ax.title.set_text(graph_name)
    fig.tight_layout()
    save_fig(fig, "task13__"+graph_name)

# 14. Побудувати кругову діаграму сумарної величини призових для кожної країни.
fig, ax = plt.subplots(figsize=(15, 10), dpi=150)
_df = df.pivot_table(index="Country", values="Career Earnings", aggfunc="sum")
_df = _df.sort_values(by="Career Earnings", ascending=False)
_df.plot(ax=ax, kind="pie", subplots=True)
ax.set_ylabel(None)
legend = ax.legend(loc="center left", bbox_to_anchor=(-0.35, 0.5), title="Career Earnings by Country")
legend_texts = legend.get_texts()
for t in legend_texts:
    new_label = f"{t.get_text()}: ${_df[_df.index == t.get_text()]['Career Earnings'].values[0]}"
    t.set_text(new_label)
save_fig(fig, "task14")

# task 15
fig, ax = plt.subplots(figsize=(10, 5))

# 15.a. Середня кількість очок для кожної країни;
mean_pts_by_country = df.groupby("Country")["Pts"].mean()
# 15.b. Середня кількість зіграних матчів тенісистами кожної країни.
mean_matches_by_country = df.groupby("Country")["Total"].mean()

_df = pd.concat([mean_matches_by_country, mean_pts_by_country], axis=1)
_df = _df.sort_values(by="Pts", ascending=False)
_df = _df.rename(columns={"Pts": "Mean pts", "Total": "Mean total matches"})
_df.plot(ax=ax, kind="bar", title="Mean pts and total matches by country")

fig.tight_layout()
save_fig(fig, "task15")

input(" Press Enter to finish ".center(71, "="))
