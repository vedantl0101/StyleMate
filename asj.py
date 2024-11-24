import pandas as pd

# Load the dataset
file_path = r'D:\ML Assignment\ML Project\StyleMate_\m4.csv'
df = pd.read_csv(file_path)

# Define valid options for each category
valid_options = {
    "Weather": ["Snowy", "Sunny", "Pleasant", "Windy"],
    "Color Palette": ["Earth", "Bright", "Pastel", "Neutral"],
    "Pattern": ["Geometric", "Floral", "Solid colors", "Stripes"],
    "Feeling": ["Unique", "Casual", "Sophisticated", "Trendy"],
    "Clothing Fit": ["Oversized", "Loose", "Standard", "Tight"]
}

# Replace invalid entries with the first valid option in each category
for column, options in valid_options.items():
    df[column] = df[column].apply(lambda x: x if x in options else options[0])

# Save the cleaned dataset
cleaned_file_path = r'D:\ML Assignment\ML Project\StyleMate_\m4_cleaned.csv'
df.to_csv(cleaned_file_path, index=False)
print(f"Cleaned dataset saved to {cleaned_file_path}")