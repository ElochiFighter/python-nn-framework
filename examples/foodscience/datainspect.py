import os
import sqlite3
import pandas as pd
from pathlib import Path


CSV_DIR = Path("examples/foodscience/data/fooddata/csvfiles")
DB_PATH = Path("examples/foodscience/database/fooddata.db")



if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

connection = sqlite3.connect(DB_PATH)



foods = pd.read_csv(
    CSV_DIR / "food.csv",
    encoding="latin1"
)

foods.to_sql(
    "food",
    connection,
    if_exists="replace",
    index=False
)



food_nutrients = pd.read_csv(
    CSV_DIR / "food_nutrient.csv",
    encoding="latin1",
    low_memory=False
)

food_nutrients.to_sql(
    "food_nutrient",
    connection,
    if_exists="replace",
    index=False
)



nutrients = pd.read_csv(
    CSV_DIR / "nutrient.csv",
    encoding="latin1"
)

nutrients.to_sql(
    "nutrient",
    connection,
    if_exists="replace",
    index=False
)



foundation_food = pd.read_csv(
    CSV_DIR / "foundation_food.csv",
    encoding="latin1",
    low_memory=False
)

foundation_food.to_sql(
    "foundation_food",
    connection,
    if_exists="replace",
    index=False
)



sample_food = pd.read_csv(
    CSV_DIR / "sample_food.csv",
    encoding="latin1"
)

sample_food.to_sql(
    "sample_food",
    connection,
    if_exists="replace",
    index=False
)



sub_sample_food = pd.read_csv(
    CSV_DIR / "sub_sample_food.csv",
    encoding="latin1"
)

sub_sample_food.to_sql(
    "sub_sample_food",
    connection,
    if_exists="replace",
    index=False
)



sub_sample_result = pd.read_csv(
    CSV_DIR / "sub_sample_result.csv",
    encoding="latin1",
    low_memory=False
)

sub_sample_result.to_sql(
    "sub_sample_result",
    connection,
    if_exists="replace",
    index=False
)



tables = pd.read_sql_query(
    """
    SELECT name
    FROM sqlite_master
    WHERE type = 'table'
    ORDER BY name;
    """,
    connection
)

print()
print("=" * 80)
print("DATABASE TABLES")
print("=" * 80)

print(
    tables.to_string(index=False)
)



print()
print("=" * 80)
print("ROW COUNTS")
print("=" * 80)

table_names = [
    "food",
    "food_nutrient",
    "nutrient",
    "foundation_food",
    "sample_food",
    "sub_sample_food",
    "sub_sample_result"
]

for table in table_names:

    result = pd.read_sql_query(
        f"SELECT COUNT(*) AS count FROM {table}",
        connection
    )

    count = result.iloc[0]["count"]

    print(f"{table:20} {count:,}")



SAMPLE_ID = 319874


query = """
SELECT
    sf.fdc_id AS sample_fdc_id,
    ssf.fdc_id AS subsample_fdc_id,

    food.description,

    fn.id AS food_nutrient_id,

    n.id AS nutrient_id,
    n.name AS nutrient_name,
    n.unit_name,

    fn.amount,

    ssr.adjusted_amount,
    ssr.lab_method_id,
    ssr.nutrient_name AS result_nutrient_name

FROM sample_food AS sf

JOIN sub_sample_food AS ssf
    ON sf.fdc_id = ssf.fdc_id_of_sample_food

JOIN food
    ON sf.fdc_id = food.fdc_id

JOIN food_nutrient AS fn
    ON ssf.fdc_id = fn.fdc_id

JOIN nutrient AS n
    ON fn.nutrient_id = n.id

LEFT JOIN sub_sample_result AS ssr
    ON fn.id = ssr.food_nutrient_id

WHERE sf.fdc_id = ?

ORDER BY
    ssf.fdc_id,
    n.name;
"""


result = pd.read_sql_query(
    query,
    connection,
    params=(SAMPLE_ID,)
)



print()
print("=" * 100)
print(f"TRACE FOR SAMPLE {SAMPLE_ID}")
print("=" * 100)

if result.empty:

    print("No data found for this sample.")

else:

    print(
        result.to_string(
            index=False
        )
    )



print()
print("=" * 100)
print("SUBSAMPLES")
print("=" * 100)

subsamples = result[
    [
        "sample_fdc_id",
        "subsample_fdc_id"
    ]
].drop_duplicates()

print(
    subsamples.to_string(
        index=False
    )
)



print()
print("=" * 100)
print("NUTRIENT SUMMARY")
print("=" * 100)

summary = (
    result[
        [
            "nutrient_name",
            "unit_name",
            "amount",
            "adjusted_amount"
        ]
    ]
    .drop_duplicates()
    .sort_values("nutrient_name")
)

print(
    summary.to_string(
        index=False
    )
)



print()
print("=" * 100)
print("SAMPLE STATISTICS")
print("=" * 100)

print(
    f"Total nutrient measurements: "
    f"{len(result)}"
)

print(
    f"Total subsamples: "
    f"{result['subsample_fdc_id'].nunique()}"
)

# ============================================================
# NUTRIENT COVERAGE
# ============================================================

print()
print("=" * 100)
print("NUTRIENT COVERAGE")
print("=" * 100)

coverage_query = """
SELECT
    n.id AS nutrient_id,
    n.name AS nutrient_name,
    n.unit_name,

    COUNT(DISTINCT sf.fdc_id) AS sample_count

FROM sample_food AS sf

JOIN sub_sample_food AS ssf
    ON sf.fdc_id = ssf.fdc_id_of_sample_food

JOIN food_nutrient AS fn
    ON ssf.fdc_id = fn.fdc_id

JOIN nutrient AS n
    ON fn.nutrient_id = n.id

GROUP BY
    n.id,
    n.name,
    n.unit_name

ORDER BY
    sample_count DESC;
"""

coverage = pd.read_sql_query(
    coverage_query,
    connection
)

print(
    coverage.to_string(index=False)
)

# ============================================================
# COMPLETE FEATURE COVERAGE
# ============================================================

print()
print("=" * 100)
print("COMPLETE FEATURE COVERAGE")
print("=" * 100)


features = [
    "Water",
    "Ash",
    "Total lipid (fat)",
    "Calcium, Ca",
    "Iron, Fe",
    "Magnesium, Mg",
    "Manganese, Mn",
    "Phosphorus, P",
    "Potassium, K",
    "Sodium, Na",
    "Zinc, Zn",
    "Nitrogen"
]


placeholders = ",".join("?" for _ in features)


complete_query = f"""
SELECT
    COUNT(*) AS complete_samples

FROM (

    SELECT
        sf.fdc_id

    FROM sample_food AS sf

    JOIN sub_sample_food AS ssf
        ON sf.fdc_id = ssf.fdc_id_of_sample_food

    JOIN food_nutrient AS fn
        ON ssf.fdc_id = fn.fdc_id

    JOIN nutrient AS n
        ON fn.nutrient_id = n.id

    WHERE n.name IN ({placeholders})

    GROUP BY sf.fdc_id

    HAVING COUNT(DISTINCT n.name) = ?

);
"""


complete_result = pd.read_sql_query(
    complete_query,
    connection,
    params=features + [len(features)]
)


print(
    complete_result.to_string(index=False)
)

# ============================================================
# BUILD ML DATASET
# ============================================================

print()
print("=" * 100)
print("BUILDING ML DATASET")
print("=" * 100)

features = [
    "Water",
    "Ash",
    "Total lipid (fat)",
    "Calcium, Ca",
    "Iron, Fe",
    "Magnesium, Mg",
    "Manganese, Mn",
    "Phosphorus, P",
    "Potassium, K",
    "Sodium, Na",
    "Zinc, Zn",
    "Nitrogen"
]

placeholders = ",".join("?" for _ in features)

ml_query = f"""
SELECT
    sf.fdc_id,
    f.description,
    n.name AS nutrient_name,
    fn.amount

FROM sample_food AS sf

JOIN food AS f
    ON sf.fdc_id = f.fdc_id

JOIN sub_sample_food AS ssf
    ON sf.fdc_id = ssf.fdc_id_of_sample_food

JOIN food_nutrient AS fn
    ON ssf.fdc_id = fn.fdc_id

JOIN nutrient AS n
    ON fn.nutrient_id = n.id

WHERE n.name IN ({placeholders});
"""

ml_long = pd.read_sql_query(
    ml_query,
    connection,
    params=features
)

print("Long dataset:")
print(ml_long.head())
print(ml_long.shape)

ml_wide = ml_long.pivot_table(
    index=["fdc_id", "description"],
    columns="nutrient_name",
    values="amount",
    aggfunc="mean"
).reset_index()

ml_wide = ml_wide.dropna(
    subset=features
)

print()
print("Wide ML dataset:")
print(ml_wide.head())

print()
print("Shape:")
print(ml_wide.shape)

print()
print("Columns:")
print(ml_wide.columns)

connection.commit()
connection.close()

print()
print("=" * 80)
print("DATABASE IMPORT COMPLETE")
print("=" * 80)