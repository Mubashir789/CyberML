import re
import csv
import joblib  # Import joblib to load saved models
import os 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

files_to_remove = [
    'New Abnormals.csv',
    'prediction_counts_barplot.png',
    'Types of Abnormal URLs.png',
    'url_predictions.csv',
    'prediction_distribution_piechart.png'
]

for file in files_to_remove:
    if os.path.exists(file):
        os.remove(file)
        print(f"Removed: {file}")
    else:
        print(f"File not found: {file}")
# Load the saved vectorizer and model
vectorizer = joblib.load('count_vectorizer.pkl')
model = joblib.load('random_forest_model.pkl')

# Function to test a new URL
def test_url(new_url):
    # Transform the new URL into the same feature space
    new_url_transformed = vectorizer.transform([new_url])
    
    # Predict the label for the new URL
    prediction = model.predict(new_url_transformed)
    
    # Return the prediction result
    return "Abnormal" if prediction == 1 else "Normal"

# Regular expression pattern to match URLs
url_pattern = re.compile(r'http://[^\s"]+')

# Initialize an empty list to store the lines
lines_list = []

# Open and read the file line by line
with open('/var/log/apache2/access.log', 'r') as file:
    for line in file:
        # Strip any leading/trailing whitespace (including newlines) and add to the list
        lines_list.append(line.strip())

# Open a CSV file to write the results
with open('url_predictions.csv', 'w', newline='') as csvfile:
    # Create a CSV writer object
    csv_writer = csv.writer(csvfile)
    
    # Write the header row
    csv_writer.writerow(['URL', 'Prediction'])
    
    # Process each line and write results to CSV
    for every_line in lines_list:
        urls = url_pattern.findall(every_line)
        for url in urls:
            result = test_url(url)
            # Write the URL and its prediction to the CSV file
            csv_writer.writerow([url, result])

print("Results have been written to 'url_predictions.csv'.")
# Count the number of each prediction type
# Load the CSV file into a DataFrame
df = pd.read_csv('url_predictions.csv')

prediction_counts = df['Prediction'].value_counts()

# Filter the DataFrame to include only rows with 'Abnormal' predictions
df_abnormal = df[df['Prediction'] == 'Abnormal']

# Save the filtered DataFrame to a new CSV file
df_abnormal.to_csv('abnormal_urls.csv', index=False)


# Create a bar plot of the prediction counts
plt.figure(figsize=(8, 6))
sns.barplot(x=prediction_counts.index, y=prediction_counts.values, palette='viridis')
plt.title('Count of URL Predictions')
plt.xlabel('Prediction')
plt.ylabel('Count')
plt.savefig('prediction_distribution_piechart.png')

# Optional: Create a pie chart of the prediction distribution
plt.figure(figsize=(8, 8))
plt.pie(prediction_counts, labels=prediction_counts.index, autopct='%1.1f%%', colors=sns.color_palette('viridis', len(prediction_counts)))
plt.title('Distribution of URL Predictions')
plt.savefig('prediction_counts_barplot.png')

abnormals = pd.read_csv("abnormal_urls.csv")
abnormals['type'] = 'temp'

i = 0
for url in abnormals['URL']:
  # Regular expression pattern to extract 'csrf'
  pattern = re.compile(r'/vulnerabilities/(\w+)/')

  # Search for the pattern in the URL
  match = pattern.search(url)

  # Extract the matched part if available
  if match:
      var = match.group(1)
      abnormals['type'][i] = var
  i+=1

abnormals.to_csv("New Abnormals.csv")
os.remove('abnormal_urls.csv')


# Load the CSV file
df = pd.read_csv('New Abnormals.csv')

# Count the occurrences of each label in the 'type' column
label_counts = df['type'].value_counts()

# Create a bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x=label_counts.index, y=label_counts.values, palette='viridis')
plt.xlabel('Label')
plt.ylabel('Count')
plt.title('Count of Different Labels in the "type" Column')
plt.xticks(rotation=45)
plt.savefig('Types of Abnormal URLs.png')