from util.common import GPU
import time
import pandas as pd
from Chapter7.PrepareDatasetForLearning import PrepareDatasetForLearning
from Chapter7.FeatureSelection import FeatureSelectionClassification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

prepare = PrepareDatasetForLearning()

N_FORWARD_SELECTION = 50
dataset = pd.read_csv('datasets/intermediate/chapter5_result.csv')

cols = ['label_cycling', 'label_downstairs', 'label_onsubway',
        'label_playing_phone', 'label_running', 'label_standing',
        'label_upstairs', 'label_walking']

class_labels = []
# Loop over each row in the DataFrame
for index, row in dataset[cols].iterrows():
    class_label = row.idxmax()
    class_labels.append(class_label)

# Add a new column 'class' to the DataFrame with the class labels
dataset['class'] = class_labels

dataset = dataset.drop('Unnamed: 0', axis=1)
dataset = dataset.drop('time', axis=1)
dataset = dataset.dropna()

for col in dataset.columns[dataset.columns.str.contains('label')]:
    dataset = dataset.drop(col, axis=1)
for col in dataset.columns[dataset.columns.str.contains('pca')]:
    dataset = dataset.drop(col, axis=1)

X = dataset.drop('class', axis=1)
y = dataset['class']
fsc = FeatureSelectionClassification()
# splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# For Embedded method, which uesd XGBoost(or LightBoos)
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# Filter method
selected_features_filter = fsc.filter_method(20, X_train, y_train)
print("Selected features by filter method: ", selected_features_filter)

# Wrapper method
selected_features_wrapper = fsc.wrapper_method(20, X_train, y_train)
print("Selected features by wrapper method: ", selected_features_wrapper)

# Embedded method
selected_features_embedded = fsc.embedded_method(20, X_train, y_train)
print("Selected features by embedded method: ", selected_features_embedded)

print('end')
