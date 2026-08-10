import sqlite3
import pandas as pd

DB = "examples/foodscience/database/fooddata.db"

connection = sqlite3.connect(DB)

foods = pd.read_csv(
    "examples/foodscience/data/fooddata/csvfiles/food.csv",
    encoding="latin1"
)

foods.to_sql(
    "food",
    connection,
    if_exists="replace",
    index=False
)

connection.close()