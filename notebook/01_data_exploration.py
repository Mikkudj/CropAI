import pandas as pd

# Load the training dataset
train_data = pd.read_parquet(
    "data/train-00000-of-00001.parquet"
)

print("===== YIELD ANALYSIS =====")

print("\nYield statistics:")
print(train_data["Yield"].describe())

print("\nTop 10 highest yields:")
print(
    train_data[
        ["State", "Crop", "Season", "Area", "Production", "Yield"]
    ]
    .sort_values("Yield", ascending=False)
    .head(10)
)

# Check relationship between Production, Area and Yield

train_data["Calculated_Yield"] = (
    train_data["Production"] / train_data["Area"]
)

train_data["Yield_Difference"] = (
    train_data["Yield"] - train_data["Calculated_Yield"]
)

print("\n===== YIELD FORMULA CHECK =====")

print(
    train_data[
        [
            "Area",
            "Production",
            "Yield",
            "Calculated_Yield",
            "Yield_Difference"
        ]
    ].head(10)
)

print("\nCorrelation between Production and Yield:")
print(
    train_data["Production"].corr(
        train_data["Yield"]
    )
)

print("\nCorrelation between Calculated Yield and Actual Yield:")
print(
    train_data["Calculated_Yield"].corr(
        train_data["Yield"]
    )
)
print("\n===== CATEGORICAL ANALYSIS =====")

print("\nNumber of states:")
print(train_data["State"].nunique())

print("\nNumber of crops:")
print(train_data["Crop"].nunique())

print("\nNumber of seasons:")
print(train_data["Season"].nunique())

print("\nStates:")
print(train_data["State"].unique())

print("\nSeasons:")
print(train_data["Season"].unique())

print("\nTop 15 crops by number of records:")
print(train_data["Crop"].value_counts().head(15))
print("\n===== YIELD BY CROP =====")

crop_yield = (
    train_data.groupby("Crop")["Yield"]
    .agg(["count", "mean", "median"])
    .sort_values("median", ascending=False)
)

print("\nTop 15 crops by median yield:")
print(crop_yield.head(15))

print("\nBottom 15 crops by median yield:")
print(crop_yield.tail(15))
print("\n===== YIELD BY CROP =====")

crop_yield = (
    train_data.groupby("Crop")["Yield"]
    .agg(["count", "mean", "median"])
    .sort_values("median", ascending=False)
)

print("\nTop 15 crops by median yield:")
print(crop_yield.head(15))

print("\nBottom 15 crops by median yield:")
print(crop_yield.tail(15))
print("\n===== TRAIN VS TEST CATEGORIES =====")

test_data = pd.read_parquet(
    "data/test-00000-of-00001.parquet"
)

print("\nTrain shape:")
print(train_data.shape)

print("\nTest shape:")
print(test_data.shape)

print("\nStates only in test:")
print(
    set(test_data["State"].unique())
    - set(train_data["State"].unique())
)

print("\nCrops only in test:")
print(
    set(test_data["Crop"].unique())
    - set(train_data["Crop"].unique())
)

print("\nSeasons only in test:")
print(
    set(test_data["Season"].unique())
    - set(train_data["Season"].unique())
)