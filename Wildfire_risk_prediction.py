# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 10:19:00 2025

@author: Nachu
"""
#load libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

#Load data
path =  r"C:\Users\Nachu\OneDrive - University of Pittsburgh\ECON_2824\Homework\CA_wildfires.csv"
data = pd.read_csv(path)

print(data.info())

#DATA CLEANING
# Identify all object (categorical) columns
categorical_columns = data.select_dtypes(include=['object']).columns

# Standardize categorical variables
data[categorical_columns] = (
    data[categorical_columns]
    .apply(lambda col: col.str.strip().str.lower() if col.dtype == "object" else col)
    .fillna("unknown")  # Fill missing values with "unknown"
)

#standardise column names
data.columns = (
    data.columns
    .str.strip('* ')  # Remove leading '* ' or spaces
    .str.replace(r'[^\w\s]', '', regex=True)  # Remove special characters
    .str.replace(' ', '_')  # Replace spaces with underscores
    .str.lower()  # Convert to lowercase
)

print(data.columns.tolist())

#DECIDING ON FEATURES TO USE IN THE MODEL.
house_feature_colums = [
    "roof_construction",
    "eaves",
    "vent_screen",
    "exterior_siding",
    "window_pane",
    "deckporch_on_grade",
    "deckporch_elevated",
    "patio_covercarport_attached_to_structure",
    "fence_attached_to_structure",
    "distance__propane_tank_to_structure",
    "distance__residence_to_utilitymisc_structure_gt_120_sqft"
]

# Create the binary column for damage > 50% using str.contains
data["damage_dummy"] = data["damage"].str.contains("destroyed", case=False, na=False).astype(int)

# Encode categorical features as numbers using Label Encoding
encoded_features = pd.DataFrame()
for feature in house_feature_colums:
    le = LabelEncoder()
    encoded_features[feature] = le.fit_transform(data[feature])

# Calculate Mutual Information with the target (damage_dummy)
mi_scores = mutual_info_classif(encoded_features, data["damage_dummy"], discrete_features=True)

# Create a DataFrame to display results
mi_df = pd.DataFrame({
    "Feature": house_feature_colums,
    "Mutual Information": mi_scores
}).sort_values(by="Mutual Information", ascending=False)

# Display the results
print("Mutual Information Scores:")
print(mi_df)

# Calculate percentage of null/unknown for each feature
missing_percentages = {
    feature: (data[feature].replace("unknown", np.nan).isnull().mean() * 100)
    for feature in house_feature_colums if feature in data.columns
}

# Convert missing percentages to a DataFrame
missing_df = pd.DataFrame(list(missing_percentages.items()), columns=["Feature", "% Missing"])

# Merge missing percentages with Mutual Information scores
mi_df = mi_df.merge(missing_df, on="Feature", how="left")

# Sort by Mutual Information (descending) and Missing Percentage (ascending)
mi_df = mi_df.sort_values(by=["Mutual Information", "% Missing"], ascending=[False, True])

#Will select the top 2 features from mi_df for the model. They have high MI and moderate missing values.
# I will also use year built, assessed value, latitude, longitude, county along with these 2 house features.

# Define the features to use in the model
model_features = [
    "year_built_parcel", 
    "assessed_improved_value_parcel", 
    "latitude", 
    "longitude", 
    "roof_construction",
    "exterior_siding"
]

#Check missing values, zeroes in selected features
import seaborn as sns
import matplotlib.pyplot as plt

# Clean string variables
selected_features = ["exterior_siding", "roof_construction"]

for feature in selected_features:
    # Group by feature category and calculate destruction rate (>50% damage)
    feature_damage_distribution = (
        data.groupby(feature)["damage_dummy"]
        .mean()
        .reset_index()
        .sort_values(by="damage_dummy", ascending=False)
    )

    # Plot the destruction rates
    plt.figure(figsize=(8, 5))
    sns.barplot(x="damage_dummy", y=feature, data=feature_damage_distribution, palette="RdYlGn")
    plt.xlabel("Percentage of Structures Destroyed (>50% Damage)")
    plt.ylabel(feature.replace("_", " ").title())  # Format label for readability
    plt.title(f"Fire Damage Distribution by {feature.replace('_', ' ').title()}")
    plt.show()

    
#The destruction rate for "Unknown" in roof type and exterior siding is lower 
#than the most flammable materials.So roofs/sidings marked unknown are likely less flammable. 
# Dropping "Unknown" would remove useful information.So we keep it as such for our model.

#I have decided to include the following as features : Yearbuilt(limited from 1900-2024), assessed value
#county, roof construction, exterior siding,latitude, longitude.

#Deal with non string variables - assessed value and year built
# Check % missing in assessed value
missing_assessed_value = data["assessed_improved_value_parcel"].isnull().mean() * 100
print(f"Percentage of missing assessed values: {missing_assessed_value:.2f}%")

# Create missing indicator
data["assessed_value_missing"] = data["assessed_improved_value_parcel"].isnull()

# Plot damage distribution for missing vs. non-missing assessed values
plt.figure(figsize=(6, 4))
sns.barplot(x=data["assessed_value_missing"], y=data["damage_dummy"], ci=None, palette="coolwarm")
plt.xlabel("Assessed Value Missing Indicator")
plt.ylabel("Average Damage > 50%")
plt.title("Damage Distribution for Missing vs. Non-Missing Assessed Value")
plt.show()


#Since only 6.02% of the data has missing assessed_improved_value_parcel, 
#and the damage distribution difference isn't extreme, 
#I am dropping the missing values.


# Filter data for year_built_parcel < 1900 and >= 1900
pre_1900 = data[data["year_built_parcel"] < 1900]
post_1900 = data[data["year_built_parcel"] >= 1900]
missing_year_built = data[data["year_built_parcel"].isnull()]

# Calculate the percentage of damage > 50% for each group
damage_distribution_pre_1900 = pre_1900["damage_dummy"].mean() * 100
damage_distribution_post_1900 = post_1900["damage_dummy"].mean() * 100
damage_distribution_missing = missing_year_built["damage_dummy"].mean() * 100

# Create a summary DataFrame for visualization
damage_summary = pd.DataFrame({
    "Year Built Group": ["< 1900", ">= 1900", "Missing"],
    "Percentage Destroyed": [
        damage_distribution_pre_1900, 
        damage_distribution_post_1900, 
        damage_distribution_missing
    ]
})

# Plot the distribution
plt.figure(figsize=(8, 5))
sns.barplot(data=damage_summary, x="Year Built Group", y="Percentage Destroyed", palette="coolwarm")
plt.title("Fire Damage Distribution by Year Built Group")
plt.ylabel("Percentage of Structures Destroyed (>50% Damage)")
plt.xlabel("Year Built Group")
plt.ylim(0, 100)  # Set y-axis limits to percentage range
plt.show()


#Missing years fire damage aligns more with post 1900 data, so I will input median year post 1900.
median_post_1900 = post_1900["year_built_parcel"].median()
data["year_built_parcel"] = data["year_built_parcel"].fillna(median_post_1900)

# Apply filtering conditions - We now have data only from 1900-2024 with missing values with the median and any values below 
#that like 0 (around 10000 such values) are dropped.
data_model = data[
    (data["year_built_parcel"] >= 1900) & 
    (data["year_built_parcel"] <= 2024)
].dropna(subset=["assessed_improved_value_parcel"])  # Drop missing assessed values

# Select only the model features + target variable
data_model = data_model[model_features + ["damage_dummy"]]

# Display summary
print(f"Final dataset shape: {data_model.shape}")
print("Sample of data_model:")
print(data_model.head())


# MODEL  - BAYES AND KNN
# Step 1: Dummy encode categorical features
data_model = pd.get_dummies(
    data_model,
    columns=["roof_construction", "exterior_siding"],
    drop_first=True  # Drop the first category to avoid multicollinearity
)

# Step 2: Scale numerical features
numeric_features = ["year_built_parcel", "assessed_improved_value_parcel", "latitude", "longitude"]
scaler = StandardScaler()
data_model[numeric_features] = scaler.fit_transform(data_model[numeric_features])

# Step 3: Split the dataset into features (X) and target (y)
X = data_model.drop(columns=["damage_dummy"])
y = data_model["damage_dummy"]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Naïve Bayes with Cross-Validation
nb_model = GaussianNB()
nb_cv_scores = cross_val_score(nb_model, X, y, cv=5, scoring="accuracy")  # 5-fold CV

# Output Naïve Bayes cross-validation results
print("Naïve Bayes Cross-Validation Accuracy Scores:", nb_cv_scores)
print(f"Naïve Bayes Average Accuracy: {np.mean(nb_cv_scores):.2f}")

# Train and evaluate Naïve Bayes on the test set
nb_model.fit(X_train, y_train)
nb_predictions = nb_model.predict(X_test)

print("\nNaïve Bayes Test Set Evaluation:")
print(classification_report(y_test, nb_predictions))
print(f"Naïve Bayes Test Set Accuracy: {accuracy_score(y_test, nb_predictions):.2f}")

# Step 5: k-Nearest Neighbors with Hyperparameter Tuning and Cross-Validation
param_grid = {"n_neighbors": range(1, 21)}  # Test k values from 1 to 20
grid_search = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)
grid_search.fit(X, y)

# Output the best k and its corresponding cross-validation accuracy
best_k = grid_search.best_params_["n_neighbors"]
best_accuracy = grid_search.best_score_
print(f"\nBest k for kNN: {best_k}")
print(f"Best Cross-Validated Accuracy for kNN: {best_accuracy:.2f}")

# Train and evaluate kNN on the test set using the best k
knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_train, y_train)
knn_predictions = knn_model.predict(X_test)

print("\nkNN Test Set Evaluation:")
print(classification_report(y_test, knn_predictions))
print(f"kNN Test Set Accuracy: {accuracy_score(y_test, knn_predictions):.2f}")

# Step 6: Confusion Matrix for kNN
conf_matrix = confusion_matrix(y_test, knn_predictions)
print("\nConfusion Matrix for kNN:")
print(conf_matrix)

# kNN Model Performs better than Bayes.

# Accuracy: 88%
# The model correctly classifies 88% of all cases as damaged or not damaged.

# Precision:
# - 89% (damage): When the model predicts a home is damaged, it is correct 89% of the time.
# - 85% (no damage): When the model predicts no damage, it is correct 85% of the time.

# Recall:
# - 89% (damage): The model captures 89% of all truly damaged homes but misses 11% (false negatives).
# - 86% (no damage): The model correctly identifies 86% of non-damaged homes but misclassifies 14% as damaged.

# Confusion Matrix Insights:
# - False negatives (missed damaged homes): 1,602 cases (11% of truly damaged homes).
# - False positives (wrongly flagged as damaged): 1,507 cases (14% of non-damaged homes).
# - More false negatives than false positives, meaning the model misses more damaged homes than it wrongly flags.

# The false negatives are higher than false positives, so the model could be fine-tuned to improve recall 
# and better capture truly damaged homes.



# Step 7:  Scatter Plot of kNN Predictions (Geographic Visualization)
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
for damage_level, color in zip([0, 1], ["green", "red"]):
    subset = X_test[(knn_predictions == damage_level)]
    plt.scatter(subset["longitude"], subset["latitude"], c=color, label=f"Damage > 50%: {damage_level}", alpha=0.6)

plt.title("Map of Predicted Damage Levels (kNN)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.show()

#Houses around latitude 38 to 39 ans 120 to 121 seem to have sustained lesser damage compared to other areas. 
#Being a little westward in the same latitude equals more fire damage.  


#Additional insights
#Cleaning other columns 
# State column is redundant, we know it is California, so can drop it.
data = data.drop(columns=["state"])

# Drop 'Community' column - Community does not add any additional insight
data = data.drop(columns=["community"])

# There are about 50 counties, far more manageable than cities, so can focus on cleaning counties first
# Find rows where both 'County' and 'City' are missing
missing_city_county = data[data["city"].isnull() & data["county"].isnull()]
# No values have both city and county missing. If we know the city, we can figure out the county.

# Identify rows with missing county but available city
missing_county_rows = data[data["county"].isnull() & data["city"].notnull()]
print(f"Number of rows with missing county but available city: {len(missing_county_rows)}")

# Figure out the county for each city (something like Excel's pivot table)
city_county_df = (
    data.dropna(subset=["county"])  # Exclude rows where county is null
    .drop_duplicates(subset=["city"])  # Ensure each city is listed only once
    .loc[:, ["city", "county"]]  # Select only city and county columns
    .sort_values(by="city")  # Sort for readability
    .reset_index(drop=True)  # Reset index for cleaner presentation
)

# Fill missing counties in the original data using the city_county_df mapping
data["county"] = data.apply(
    lambda row: city_county_df[city_county_df["city"] == row["city"]]["county"].values[0]
    if pd.isnull(row["county"]) and row["city"] in city_county_df["city"].values
    else row["county"],
    axis=1,
)

# Check if all missing counties are filled
remaining_missing_counties = data[data["county"].isnull()]
print(f"Number of rows with missing counties after imputation: {len(remaining_missing_counties)}")

if not remaining_missing_counties.empty:
    print("Cities with missing counties:")
    print(remaining_missing_counties["city"].unique())
else:
    print("All missing counties have been successfully filled!")

# Brownsville does not have a county name anywhere in the dataset.
# Since it's just one city with a missing county, we can google its county to input the value manually.
# Brownsville is in Yuba County.

# Fill all missing counties with the name 'Yuba'
data["county"] = data["county"].fillna("Yuba")

# Check if there are any remaining missing counties
remaining_missing_counties = data[data["county"].isnull()]
print(f"Number of rows with missing counties after filling with 'Yuba': {len(remaining_missing_counties)}")

# City names are unclean, with a lot more missing values than counties, and values like "A".
# So we will focus on analyzing counties. However, we will not drop the city column from data,
# later can clean and use it for further analysis if necessary.

#Visualisation of damaged>50% by county
# Group by county and calculate destruction rates
county_damage_distribution = (
    data.groupby("county")["damage_dummy"]
    .mean()
    .reset_index()
    .sort_values(by="damage_dummy", ascending=False)
)

# Add latitude and longitude for plotting
data_county_centroids = data.dropna(subset=["county"]).groupby("county")[["latitude", "longitude"]].mean().reset_index()
county_damage_distribution = county_damage_distribution.merge(data_county_centroids, on="county")

# Scatter plot with improved visualization
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=county_damage_distribution, 
    x="longitude", 
    y="latitude", 
    size="damage_dummy", 
    hue="damage_dummy", 
    palette="YlOrRd",  # Use neutral tones for circles
    legend=False, 
    sizes=(50, 500)  # Scale circle sizes
)

# Add county names with dynamic transparency based on damage rate
for i, row in county_damage_distribution.iterrows():
    plt.text(
        row["longitude"], 
        row["latitude"], 
        row["county"], 
        fontsize=8, 
        color="black",  # Black text for clarity
        alpha=min(1, row["damage_dummy"] + 0.3)  # Transparency increases for lesser-affected counties
    )

plt.title("Fire Damage Rates by County (Scatter Plot)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
print(data["county"].unique()) 


# Plot only the top 10 counties
plt.figure(figsize=(12, 6))
sns.barplot(data=county_damage_distribution.head(10), x="damage_dummy", y="county", palette="coolwarm")
plt.xlabel("Destruction Rate (>50% Damage)")
plt.ylabel("County")
plt.title("Top 10 Counties with Highest Fire Damage Rates")
plt.show()
