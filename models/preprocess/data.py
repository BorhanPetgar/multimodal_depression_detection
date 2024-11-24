import pandas as pd

# Read the CSV files
df1 = pd.read_csv('/content/textual_dataset_happy.csv')
df2 = pd.read_csv('/content/textual_dataset_sad.csv')

# Add a column with zeros to the first DataFrame
df1['label'] = 0

# Add a column with ones to the second DataFrame
df2['label'] = 1


# Merge the two DataFrames
merged_df = pd.concat([df1, df2])

# Shuffle the merged DataFrame
shuffled_df = merged_df.sample(frac=1).reset_index(drop=True)

# Save the shuffled DataFrame to a new CSV file
shuffled_df.to_csv('merged_shuffled.csv', index=False) 
