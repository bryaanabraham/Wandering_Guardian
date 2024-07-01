# Wander Guardian: AI-Driven Healthcare Solutions for Dementia Patients

## Overview

This project aims to provide AI-driven healthcare solutions for dementia patients by performing navigational analysis using GPS data. The code identifies whether the patient is on a known safe path or if they seem to be deviating drastically from the path, indicating a potential dementia attack.

## Features

- **GPS Data Collection**: Collects GPS data manually from Google Maps.
- **Navigational Analysis**: Uses the collected GPS data to perform navigational analysis.
- **Safe Path Identification**: Identifies whether the patient is on a known safe path.
- **Deviation Detection**: Detects significant deviations from the safe path that may indicate a potential dementia attack.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/your-repo-name.git
    ```
2. Navigate to the project directory:
    ```sh
    cd your-repo-name
    ```
3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Load the dataset containing longitude and latitude values.
2. Train the model for path prediction using the provided GPS data.
3. Perform navigational analysis to identify safe paths and detect deviations.

### Example

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import folium
from geopy.distance import geodesic
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load GPS data
data = pd.read_csv('gps_data.csv')

# Train the model
X_train, X_test, y_train, y_test = train_test_split(data[['longitude', 'latitude']], data['label'], test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Perform prediction
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Visualize the path
map = folium.Map(location=[data['latitude'].mean(), data['longitude'].mean()], zoom_start=12)
for idx, row in data.iterrows():
    folium.Marker([row['latitude'], row['longitude']], popup=row['label']).add_to(map)

map.save('path_analysis.html')
