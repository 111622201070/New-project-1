from google.colab import files
uploaded = files.upload()


import pandas as pd

# Load the dataset
file_path = "synthetic_symptom_disease.csv"  # Update with the correct path if needed
df = pd.read_csv(file_path, encoding='utf-8')

df.columns = df.columns.str.lower().str.strip()


df = df.map(lambda x: x.strip() if isinstance(x, str) else x)


df = df.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x) if col.dtype == 'object' else col)


df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

df['combined symptom'] = df['combined symptom'].apply(
    lambda x: [symptom.strip().lower() for symptom in x.split(',')] if isinstance(x, str) else []
)


df.dropna(inplace=True)

print("Cleaned Dataset Sample:")
print(df.head())

import os

# Ensure the directory exists
os.makedirs("/mnt/data", exist_ok=True)

# Save the cleaned file
cleaned_file_path = "/mnt/data/cleaned_symptom_disease.csv"
df.to_csv(cleaned_file_path, index=False)

print(f"Cleaned dataset saved to: {cleaned_file_path}")


cleaned_file_path = "/mnt/data/cleaned_symptom_disease.csv"
df.to_csv(cleaned_file_path, index=False)
print(f"Cleaned dataset saved to: {cleaned_file_path}")

import pandas as pd

# Load the cleaned CSV file
df_cleaned = pd.read_csv("/mnt/data/cleaned_symptom_disease.csv")

# Display the first few rows
df_cleaned.head()


from google.colab import files

files.download("/mnt/data/cleaned_symptom_disease.csv")


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load the dataset
file_path = "synthetic_symptom_disease.csv"  # Update path if needed
df = pd.read_csv(file_path, encoding='utf-8')

# Convert column names to lowercase & strip spaces
df.columns = df.columns.str.lower().str.strip()

# Convert 'combined symptom' column into a list of symptoms
df['combined symptom'] = df['combined symptom'].apply(
    lambda x: [symptom.strip().lower() for symptom in x.split(',')] if isinstance(x, str) else []
)

# Create a graph
G = nx.Graph()

# Add nodes & edges
for _, row in df.iterrows():
    disease = row['disease']
    symptoms = row['combined symptom']

    G.add_node(disease, type='disease', color='red')  # Add disease node
    for symptom in symptoms:
        G.add_node(symptom, type='symptom', color='blue')  # Add symptom node
        G.add_edge(disease, symptom)  # Connect disease to symptom

# Assign node colors for visualization
node_colors = ["red" if G.nodes[n]['type'] == 'disease' else "blue" for n in G.nodes]

# Draw the network
plt.figure(figsize=(14, 10))
pos = nx.spring_layout(G, k=0.3, seed=42)  # Layout for better spacing
nx.draw(G, pos, with_labels=True, node_size=1000, font_size=8, node_color=node_colors, edge_color='gray')

plt.title("Semantic Network of Symptoms and Diseases")
plt.show()


import pandas as pd

# Load the cleaned dataset
file_path = "/mnt/data/cleaned_symptom_disease.csv"
df = pd.read_csv(file_path, encoding='utf-8')

# Convert 'combined symptom' column from string to list
df['combined symptom'] = df['combined symptom'].apply(eval)

# Extract unique symptoms
all_symptoms = set(symptom for symptoms in df['combined symptom'] for symptom in symptoms)

# Create a feature matrix
feature_data = []
for _, row in df.iterrows():
    disease = row['disease']
    symptoms = row['combined symptom']

    feature_vector = {symptom: (1 if symptom in symptoms else 0) for symptom in all_symptoms}
    feature_vector['disease'] = disease
    feature_data.append(feature_vector)

# Convert to DataFrame
feature_df = pd.DataFrame(feature_data)

# Save feature matrix
feature_df.to_csv("/mnt/data/feature_matrix.csv", index=False)
print("Feature matrix saved as CSV.")


df['symptom_text'] = df['symptom_text'].fillna('')  # Replace NaNs with empty strings


print(df['symptom_text'].head())  # View first few rows
print(df['symptom_text'].isna().sum())  # Count NaN values
print(df['symptom_text'].apply(lambda x: len(str(x).strip()) == 0).sum())  # Count empty strings


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample DataFrame
data = {'symptom_text': ['Fever and chills', 'Cough', 'Shortness of breath', 'Sore throat', 'Headache']}
df = pd.DataFrame(data)

# Initialize the TF-IDF Vectorizer (stop_words=None means no words will be removed)
vectorizer = TfidfVectorizer(stop_words=None)

# Transform the 'symptom_text' column into TF-IDF vectors
X_tfidf = vectorizer.fit_transform(df['symptom_text'])

# Display shape of TF-IDF matrix
print("TF-IDF Matrix Shape:", X_tfidf.shape)

# Convert to a DataFrame for better visualization
tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
print(tfidf_df)


print(df['symptom_text'].isna().sum())  # Check for NaN values
print(df['symptom_text'].apply(lambda x: isinstance(x, str)).all())  # Ensure all are strings
print(df['symptom_text'].tolist())  # View actual data


df['symptom_text'] = df['symptom_text'].fillna('')


vectorizer = TfidfVectorizer(stop_words=None)  # Already done in your code


df['symptom_text'] = df['symptom_text'].apply(lambda x: ' '.join(x.split()))


df['symptom_text'] = df['symptom_text'].astype(str)


df['symptom_text'] = df['symptom_text'].astype(str)


from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure symptom_text column is clean
df['symptom_text'] = df['symptom_text'].fillna('').astype(str)

# Check if there are meaningful words
if df['symptom_text'].str.strip().eq('').all():
    print("Error: All symptom_text values are empty after preprocessing.")
else:
    # Apply TF-IDF
    vectorizer = TfidfVectorizer(stop_words=None)
    X_tfidf = vectorizer.fit_transform(df['symptom_text'])
    print("TF-IDF transformation successful!")


print(df.columns)  # Check if 'symptom_text' exists
print(df.shape)  # Check the number of rows and columns

# Show some sample values
print(df['symptom_text'].head(10))


print("NaN values:", df['symptom_text'].isna().sum())  # Count NaNs
print("Empty strings:", (df['symptom_text'].str.strip() == '').sum())  # Count empty strings


import pandas as pd

# Reload dataset (modify path if needed)
df = pd.read_csv("/mnt/data/cleaned_symptom_disease.csv")

# Check the data again
print(df.head())


df['symptom_text'] = df['symptom_text'].fillna('').astype(str)

if df['symptom_text'].str.strip().eq('').all():
    print("⚠️ No valid symptom text found! Check your dataset.")
else:
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words=None)
    X_tfidf = vectorizer.fit_transform(df['symptom_text'])
    print("✅ TF-IDF transformation successful!")


!pip install gensim  # Optional, for word embeddings
!pip install node2vec

import networkx as nx
import numpy as np
from node2vec import Node2Vec

# Load dataset
file_path = "/mnt/data/cleaned_symptom_disease.csv"
df = pd.read_csv(file_path, encoding='utf-8')

df['combined symptom'] = df['combined symptom'].apply(eval)

# ✅ Create a Graph
G = nx.Graph()


for _, row in df.iterrows():
    disease = row['disease']
    symptoms = row['combined symptom']

    G.add_node(disease, type='disease')  # Add disease node
    for symptom in symptoms:
        G.add_node(symptom, type='symptom')  # Add symptom node
        G.add_edge(disease, symptom)  # Connect disease to symptom


if len(G.nodes) == 0:
    print("⚠️ Error: Graph is empty! Check dataset processing.")
else:
    # ✅ Train Node2Vec Model
    node2vec = Node2Vec(G, dimensions=64, walk_length=10, num_walks=100, workers=4)
    model = node2vec.fit(window=5, min_count=1, batch_words=4)

    # ✅ Extract Node Embeddings
    embeddings = {node: model.wv[node] for node in G.nodes}

    # ✅ Save Embeddings as CSV
    embedding_df = pd.DataFrame.from_dict(embeddings, orient='index')
    embedding_df.to_csv("/mnt/data/graph_embeddings.csv")

    # ✅ Save Embeddings as NumPy File (Recommended for ML)
    np.save("/mnt/data/graph_embeddings.npy", embeddings)

    print("✅ Graph embeddings saved successfully!")

import pandas as pd

# Load the saved graph embeddings CSV file
embedding_df = pd.read_csv("/mnt/data/graph_embeddings.csv")

# Display the first few rows
print(embedding_df.head())


import numpy as np

# Load the saved NumPy file
embeddings = np.load("/mnt/data/graph_embeddings.npy", allow_pickle=True).item()

# Display some sample nodes and their embeddings
for node, embedding in list(embeddings.items())[:5]:  # Show first 5 nodes
    print(f"Node: {node}\nEmbedding: {embedding}\n")


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ✅ Load Graph Embeddings
embeddings = np.load("/mnt/data/graph_embeddings.npy", allow_pickle=True).item()

# ✅ Load Dataset
df = pd.read_csv("/mnt/data/cleaned_symptom_disease.csv")
df['combined symptom'] = df['combined symptom'].apply(eval)  # Convert string list to actual list

# ✅ Prepare Features (X) & Labels (y)
X = []
y = []

for _, row in df.iterrows():
    disease = row['disease']
    symptoms = row['combined symptom']

    # Get the embedding of the disease (if it exists)
    if disease in embeddings:
        X.append(embeddings[disease])  # Use the disease embedding as feature
        y.append(disease)  # The label is the disease itself

# ✅ Convert to NumPy Array
X = np.array(X)
y = np.array(y)

# ✅ Split Data into Training & Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# ✅ Make Predictions
y_pred = clf.predict(X_test)

# ✅ Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")
print("\n🔹 Classification Report:\n", classification_report(y_test, y_pred))

# ✅ Save Trained Model
import joblib
joblib.dump(clf, "/mnt/data/disease_classifier.pkl")
print("🎯 Trained model saved as 'disease_classifier.pkl'.")




import joblib
import numpy as np
import pandas as pd

# ✅ Load the trained model
clf = joblib.load("/mnt/data/disease_classifier.pkl")

# ✅ Load Graph Embeddings
embeddings = np.load("/mnt/data/graph_embeddings.npy", allow_pickle=True).item()

# ✅ Function to Predict Disease from Symptoms
def predict_disease(symptoms):
    symptom_vectors = []

    # Convert symptoms into embeddings
    for symptom in symptoms:
        if symptom in embeddings:
            symptom_vectors.append(embeddings[symptom])
        else:
            print(f"⚠️ Warning: '{symptom}' not found in embeddings!")

    if not symptom_vectors:
        print("❌ No valid symptoms found. Cannot predict disease.")
        return None

    # Average symptom vectors to get a single feature vector
    symptom_vector = np.mean(symptom_vectors, axis=0).reshape(1, -1)

    # Make Prediction
    predicted_disease = clf.predict(symptom_vector)[0]
    print(f"🎯 Predicted Disease: {predicted_disease}")

    return predicted_disease

# ✅ Example Test Case
test_symptoms = ["weight loss", "tremors", "vomiting"]  # Replace with real symptoms
predict_disease(test_symptoms)


# ✅ Load Graph Embeddings
import numpy as np

embeddings = np.load("/mnt/data/graph_embeddings.npy", allow_pickle=True).item()

# ✅ Print available nodes in embeddings
print("🔹 Available Nodes in Embeddings:")
print(list(embeddings.keys())[:20])  # Show first 20 nodes


# ✅ Fix 'combined symptom' column (ensure it is a list)
df['combined symptom'] = df['combined symptom'].apply(
    lambda x: [s.strip().lower() for s in eval(x)] if isinstance(x, str) else x
)


import networkx as nx

print("✅ Total Nodes in Graph:", len(G.nodes))
print("✅ Total Edges in Graph:", len(G.edges))


import pandas as pd

# ✅ Load the dataset
df = pd.read_csv("/mnt/data/cleaned_symptom_disease.csv")

# ✅ Fix 'combined symptom' column (convert string to list)
df['combined symptom'] = df['combined symptom'].apply(lambda x: eval(x) if isinstance(x, str) else x)

# ✅ Check if symptoms are now in list format
print(df.head())


from node2vec import Node2Vec

# ✅ Create a new graph
G = nx.Graph()

# ✅ Add disease-symptom relationships
for _, row in df.iterrows():
    disease = row['disease']
    symptoms = row['combined symptom']

    if isinstance(symptoms, list):  # Ensure it's a list
        for symptom in symptoms:
            G.add_edge(disease, symptom)

# ✅ Check Graph
print("✅ Total Nodes in Graph:", len(G.nodes))
print("✅ Total Edges in Graph:", len(G.edges))

# 🚀 Train Node2Vec model (ONLY if the graph is not empty)
if len(G.nodes) > 0:
    node2vec = Node2Vec(G, dimensions=64, walk_length=10, num_walks=100, workers=4)
    model = node2vec.fit(window=5, min_count=1, batch_words=4)

    # ✅ Extract embeddings
    embeddings = {node: model.wv[node] for node in G.nodes}
    np.save("/mnt/data/graph_embeddings.npy", embeddings)

    print("✅ New Graph Embeddings Saved!")
else:
    print("❌ Error: Graph is empty! Fix the dataset before training.")


import pandas as pd

# ✅ Load the dataset
df = pd.read_csv("/mnt/data/cleaned_symptom_disease.csv")

# ✅ Check the first few rows
print(df.head())

# ✅ Check data types
print(df.dtypes)


# ✅ Convert symptom_text into a list
df['combined symptom'] = df['symptom_text'].apply(lambda x: x.lower().split() if isinstance(x, str) else [])

# ✅ Check if symptoms are correctly stored as lists
print(df[['disease', 'combined symptom']].head())


import networkx as nx
from node2vec import Node2Vec
import numpy as np

# ✅ Create the graph
G = nx.Graph()

# ✅ Add disease-symptom relationships
for _, row in df.iterrows():
    disease = row['disease']
    symptoms = row['combined symptom']

    if isinstance(symptoms, list) and len(symptoms) > 0:  # Ensure symptoms are in list format
        for symptom in symptoms:
            G.add_edge(disease, symptom)

# ✅ Check Graph
print("✅ Total Nodes in Graph:", len(G.nodes))
print("✅ Total Edges in Graph:", len(G.edges))

# 🚀 Train Node2Vec only if the graph is not empty
if len(G.nodes) > 0:
    node2vec = Node2Vec(G, dimensions=64, walk_length=10, num_walks=100, workers=4)
    model = node2vec.fit(window=5, min_count=1, batch_words=4)

    # ✅ Save embeddings
    embeddings = {node: model.wv[node] for node in G.nodes}
    np.save("/mnt/data/graph_embeddings.npy", embeddings)

    print("✅ New Graph Embeddings Saved!")
else:
    print("❌ Error: Graph is still empty! Check dataset formatting again.")


# ✅ Convert symptom_text into a list of symptoms
df['combined symptom'] = df['symptom_text'].apply(lambda x: x.lower().split() if isinstance(x, str) else [])

# ✅ Check if symptoms are now stored as lists
print(df[['disease', 'combined symptom']].head())


import networkx as nx
from node2vec import Node2Vec
import numpy as np

# ✅ Create the graph
G = nx.Graph()

# ✅ Add disease-symptom relationships
for _, row in df.iterrows():
    disease = row['disease']
    symptoms = row['combined symptom']

    if isinstance(symptoms, list) and len(symptoms) > 0:  # Ensure symptoms are in list format
        for symptom in symptoms:
            G.add_edge(disease, symptom)

# ✅ Check Graph
print("✅ Total Nodes in Graph:", len(G.nodes))
print("✅ Total Edges in Graph:", len(G.edges))

# 🚀 Train Node2Vec only if the graph is not empty
if len(G.nodes) > 0:
    node2vec = Node2Vec(G, dimensions=64, walk_length=10, num_walks=100, workers=4)
    model = node2vec.fit(window=5, min_count=1, batch_words=4)

    # ✅ Save embeddings
    embeddings = {node: model.wv[node] for node in G.nodes}
    np.save("/mnt/data/graph_embeddings.npy", embeddings)

    print("✅ New Graph Embeddings Saved!")
else:
    print("❌ Error: Graph is still empty! Check dataset formatting again.")


# Check for missing values
print(df.isnull().sum())

# Drop rows where 'disease' or 'symptom_text' is missing
df = df.dropna(subset=['disease', 'symptom_text'])

print("✅ Missing values removed!")


print(df.dtypes)

# Convert 'combined symptom' to a proper list if needed
if isinstance(df['combined symptom'].iloc[0], str):
    df['combined symptom'] = df['combined symptom'].apply(lambda x: x.lower().split() if isinstance(x, str) else [])

print("✅ Data types verified!")


# Remove rows where 'combined symptom' is empty
df = df[df['combined symptom'].apply(lambda x: len(x) > 0)]

print("✅ Empty symptom lists removed!")


import networkx as nx

# Create a quick check graph
G_test = nx.Graph()

for _, row in df.iterrows():
    disease = row['disease']
    symptoms = row['combined symptom']

    for symptom in symptoms:
        G_test.add_edge(disease, symptom)

# Check graph properties
print(f"✅ Total Nodes: {len(G_test.nodes)}")
print(f"✅ Total Edges: {len(G_test.edges)}")

if len(G_test.nodes) == 0:
    print("❌ Error: No valid connections found! Check dataset.")


from sklearn.feature_extraction.text import TfidfVectorizer

# ✅ Convert symptoms into text format for vectorization
df['symptom_text'] = df['combined symptom'].apply(lambda x: ' '.join(x))

# ✅ Apply TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['symptom_text'])

print("✅ TF-IDF feature extraction completed!")


import numpy as np

# ✅ Load the pre-trained Node2Vec embeddings
embeddings = np.load("/mnt/data/graph_embeddings.npy", allow_pickle=True).item()

# ✅ Convert symptoms into numerical vectors
X = np.array([np.mean([embeddings.get(symptom, np.zeros(64)) for symptom in symptoms], axis=0) for symptoms in df['combined symptom']])

print("✅ Node2Vec embeddings extracted!")


from sklearn.preprocessing import LabelEncoder

# ✅ Encode disease labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['disease'])

print("✅ Disease labels encoded!")


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ✅ Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# ✅ Predict on test set
y_pred = model.predict(X_test)

# ✅ Check accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")


import joblib

# ✅ Save the model
joblib.dump(model, "/mnt/data/disease_prediction_model.pkl")
joblib.dump(vectorizer, "/mnt/data/tfidf_vectorizer.pkl")
joblib.dump(label_encoder, "/mnt/data/label_encoder.pkl")

print("✅ Model and vectorizer saved!")


# ✅ Load the trained TF-IDF vectorizer
vectorizer = joblib.load("/mnt/data/tfidf_vectorizer.pkl")

# ✅ Convert symptoms to vector using the SAME TF-IDF model
new_X = vectorizer.transform(["fever headache cough"])  # Example input

# ✅ Check feature size
print(f"New X shape: {new_X.shape}, Expected: {model.n_features_in_}")


import numpy as np

# ✅ Load the embeddings
embeddings = np.load("/mnt/data/graph_embeddings.npy", allow_pickle=True).item()

# ✅ Convert input symptoms to the correct embedding format
new_symptoms = ["fever", "headache", "cough"]
new_X = np.array([np.mean([embeddings.get(symptom, np.zeros(64)) for symptom in new_symptoms], axis=0)])

# ✅ Check feature size
print(f"New X shape: {new_X.shape}, Expected: {model.n_features_in_}")


print(f"Shape of training data (X): {X.shape}")  # Should be (num_samples, 64)
print(f"Number of features expected by model: {model.n_features_in_}")  # Should match 64


import numpy as np

# ✅ Load the trained Node2Vec embeddings
embeddings = np.load("/mnt/data/graph_embeddings.npy", allow_pickle=True).item()

# ✅ Convert input symptoms into the same 64-dimensional format
new_symptoms = ["fever", "headache", "cough"]
new_X = np.array([np.mean([embeddings.get(symptom, np.zeros(64)) for symptom in new_symptoms], axis=0)])

# ✅ Check shape before making predictions
print(f"Shape of new input data: {new_X.shape}")


from node2vec import Node2Vec
import networkx as nx

# ✅ Create the graph
G = nx.Graph()
for _, row in df.iterrows():
    disease = row['disease']
    for symptom in row['combined symptom']:
        G.add_edge(disease, symptom)

# ✅ Train Node2Vec model
node2vec = Node2Vec(G, dimensions=64, walk_length=10, num_walks=100, workers=4)
model = node2vec.fit(window=5, min_count=1, batch_words=4)

# ✅ Extract embeddings and save them
embeddings = {node: model.wv[node] for node in G.nodes}
np.save("/mnt/data/graph_embeddings.npy", embeddings)

print("✅ Node2Vec embeddings re-trained and saved!")


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np

# ✅ Load the trained Node2Vec embeddings
embeddings = np.load("/mnt/data/graph_embeddings.npy", allow_pickle=True).item()

# ✅ Extract features (X) and labels (y)
X = np.array([embeddings[node] for node in embeddings.keys() if node in df['disease'].values])
y = np.array([node for node in embeddings.keys() if node in df['disease'].values])

# ✅ Encode disease labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# ✅ Train a classifier (Logistic Regression)
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X, y_encoded)

print("✅ Classifier trained successfully!")


# ✅ Convert symptoms into 64-dimensional embeddings
new_symptoms = ["fever", "headache", "cough"]
new_X = np.array([np.mean([embeddings.get(symptom, np.zeros(64)) for symptom in new_symptoms], axis=0)])

# ✅ Ensure new_X has the correct shape
if new_X.shape == (1, 64):
    predicted_disease = classifier.predict(new_X)
    predicted_label = label_encoder.inverse_transform(predicted_disease)
    print(f"🩺 Predicted Disease: {predicted_label[0]}")
else:
    print("❌ Error: New input data is not in the correct shape!")


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ✅ Load embeddings
embeddings = np.load("/mnt/data/graph_embeddings.npy", allow_pickle=True).item()

# ✅ Extract features (X) and labels (y)
X = np.array([embeddings[node] for node in embeddings.keys() if node in df['disease'].values])
y = np.array([node for node in embeddings.keys() if node in df['disease'].values])

# ✅ Encode disease labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# ✅ Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# ✅ Train the classifier
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train)

print("✅ Model training complete!")


# ✅ Make predictions on the test data
y_pred = classifier.predict(X_test)

# ✅ Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"📊 Model Accuracy: {accuracy:.4f}")

# ✅ Generate classification report
print("\n🔍 Classification Report:\n", classification_report(y_test, y_pred))

# ✅ Display confusion matrix
print("\n🛠 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# ✅ Convert symptoms into 64-dimensional embeddings
new_symptoms = ["fever", "headache", "cough"]
new_X = np.array([np.mean([embeddings.get(symptom, np.zeros(64)) for symptom in new_symptoms], axis=0)])

# ✅ Ensure new_X has the correct shape
if new_X.shape == (1, 64):
    predicted_disease = classifier.predict(new_X)
    predicted_label = label_encoder.inverse_transform(predicted_disease)
    print(f"🩺 Predicted Disease: {predicted_label[0]}")
else:
    print("❌ Error: New input data is not in the correct shape!")


# ✅ Check if the model is trained
if hasattr(classifier, "coef_"):
    print("✅ Model is trained and ready!")
else:
    print("❌ Model is not trained! Train it first.")


from sklearn.metrics import accuracy_score

# ✅ Make predictions on test data
y_pred = classifier.predict(X_test)

# ✅ Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"📊 Model Accuracy: {accuracy:.4f}")

# If accuracy is too low, retrain the model with better features or more data.


import numpy as np

# ✅ Example new symptom input
new_symptoms = ["fever", "headache", "cough"]

# ✅ Convert symptoms into embeddings
new_X = np.array([np.mean([embeddings.get(symptom, np.zeros(64)) for symptom in new_symptoms], axis=0)])

# ✅ Ensure the input shape is correct before making predictions
if new_X.shape == (1, 64):
    predicted_disease = classifier.predict(new_X)
    predicted_label = label_encoder.inverse_transform(predicted_disease)
    print(f"🩺 Predicted Disease: {predicted_label[0]}")
else:
    print("❌ Error: Incorrect input shape!")


import numpy as np

def predict_disease(user_symptoms, classifier, label_encoder, embeddings):
    """
    Predicts a disease based on user input symptoms using the trained classifier.
    """
    # ✅ Convert symptoms into embeddings (handle unknown symptoms)
    symptom_vectors = [embeddings.get(symptom, np.zeros(64)) for symptom in user_symptoms]

    # ✅ Compute the average vector of all symptoms
    if len(symptom_vectors) > 0:
        new_X = np.mean(symptom_vectors, axis=0).reshape(1, -1)
    else:
        print("❌ Error: No valid symptoms found in the dataset!")
        return

    # ✅ Ensure correct feature size before prediction
    if new_X.shape[1] == 64:
        predicted_probabilities = classifier.predict_proba(new_X)  # Get probability scores
        predicted_index = np.argmax(predicted_probabilities)  # Get the highest probability disease
        predicted_label = label_encoder.inverse_transform([predicted_index])[0]

        print(f"🩺 Predicted Disease: {predicted_label}")
        print(f"📊 Confidence Score: {predicted_probabilities[0][predicted_index]:.4f}")

    else:
        print("❌ Error: Incorrect input shape!")



# Example: Input symptoms
user_symptoms = ["fever", "headache","itching"]  # Modify as needed

# Call the function to predict disease
predict_disease(user_symptoms, classifier, label_encoder, embeddings)


import numpy as np

def predict_disease(user_symptoms, classifier, label_encoder, embeddings):
    """
    Predicts a disease based on user input symptoms using the trained classifier.
    Only considers valid symptoms that exist in the trained embeddings.
    """
    valid_symptoms = [symptom for symptom in user_symptoms if symptom in embeddings]

    if not valid_symptoms:
        print("❌ Error: None of the input symptoms are in the dataset!")
        return

    # ✅ Convert valid symptoms to embeddings
    symptom_vectors = [embeddings[symptom] for symptom in valid_symptoms]

    # ✅ Compute the mean vector (final input)
    new_X = np.mean(symptom_vectors, axis=0).reshape(1, -1)

    # ✅ Ensure correct feature size before prediction
    if new_X.shape[1] == 64:
        predicted_probabilities = classifier.predict_proba(new_X)  # Get probability scores
        predicted_index = np.argmax(predicted_probabilities)  # Get the highest probability disease
        predicted_label = label_encoder.inverse_transform([predicted_index])[0]

        print(f"🩺 Predicted Disease: {predicted_label}")
        print(f"📊 Confidence Score: {predicted_probabilities[0][predicted_index]:.4f}")

    else:
        print("❌ Error: Incorrect input shape!")



import numpy as np

def predict_disease_with_confidence(user_symptoms, model, label_encoder, embeddings):
    """
    Predicts the most likely disease based on user-provided symptoms and gives a confidence score.

    Parameters:
    - user_symptoms: List of symptoms provided by the user.
    - model: Trained classification model (e.g., Logistic Regression).
    - label_encoder: Encoder used to convert disease labels.
    - embeddings: Node2Vec-trained symptom embeddings.

    Returns:
    - Predicted disease with confidence score or an error message.
    """
    valid_symptoms = [symptom for symptom in user_symptoms if symptom in embeddings]

    if not valid_symptoms:
        print("❌ No valid symptoms found in embeddings! Cannot predict disease.")
        return None

    # Create a feature vector by averaging the embeddings of valid symptoms
    symptom_vectors = np.array([embeddings[symptom] for symptom in valid_symptoms])
    user_vector = np.mean(symptom_vectors, axis=0).reshape(1, -1)

    # Ensure the input feature dimension matches the trained model
    if user_vector.shape[1] != model.n_features_in_:
        print(f"⚠️ Mismatch in feature dimensions! Expected {model.n_features_in_}, but got {user_vector.shape[1]}.")
        return None

    # Predict the disease with probability
    probabilities = model.predict_proba(user_vector)  # Get probability of each disease
    top_disease_index = np.argmax(probabilities)  # Find the most probable disease
    top_disease = label_encoder.inverse_transform([top_disease_index])[0]  # Get disease name
    confidence = probabilities[0][top_disease_index] * 100  # Convert to percentage

    print(f"🩺 **Predicted Disease:** {top_disease} (Confidence: {confidence:.2f}%)")
    return top_disease, confidence

# ✅ Example User Input
user_symptoms = ["memoryloss", "depression", "chills"]  # Replace with any symptoms

# ✅ Run Prediction
predicted_disease, confidence = predict_disease_with_confidence(user_symptoms, classifier, label_encoder, embeddings)


print("X shape:", X.shape)
print("y shape:", y.shape)
print("X empty?", X.empty if hasattr(X, "empty") else len(X) == 0)
print("y empty?", y.empty if hasattr(y, "empty") else len(y) == 0)


print(X[:5])  # Check the first few rows
print(y[:5])  # Check target values


print(df.shape)  # Check dataset size
print(df.isna().sum())  # Count missing values


print(df.columns)


# Define features and target
X = df[['combined symptom']]  # Use actual feature column(s)
y = df['disease_encoded']  # Use encoded labels

# Verify data before splitting
print("Feature sample:\n", X.head())
print("Target sample:\n", y.head())

# Split Data
from sklearn.model_selection import train_test_split

if X.empty or y.empty:
    print("⚠️ Error: No valid data for training. Check dataset!")
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("✅ Data successfully split!")


print("✅ Total Samples:", len(X))


X = []
y = []

for _, row in df.iterrows():
    embedding = get_symptom_embedding(row["combined symptom"], embeddings)
    if embedding is not None:  # Ensure valid embeddings exist
        X.append(embedding)
        y.append(row["disease_encoded"])

X = np.array(X)
y = np.array(y)

print("✅ Final Feature Matrix Shape:", X.shape)
print("✅ Number of Labels:", len(y))


print("✅ Total Nodes in Graph:", len(embeddings))
print("✅ Sample Embedding:", embeddings.get("fever", "⚠️ Not Found"))


print("✅ Total Nodes in Embeddings:", len(embeddings))
print("✅ Example Embedding for 'fever':", embeddings.get("fever", "⚠️ Not Found"))


import ast
df["combined symptom"] = df["combined symptom"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
print("✅ Sample Symptoms:", df["combined symptom"].head())


def get_symptom_embedding(symptoms, embeddings):
    valid_symptoms = [symptom for symptom in symptoms if symptom in embeddings]
    if not valid_symptoms:
        print(f"⚠️ No valid embeddings for: {symptoms}")
        return None
    symptom_vectors = np.array([embeddings[symptom] for symptom in valid_symptoms])
    return np.mean(symptom_vectors, axis=0)

# ✅ Check first 5 rows
for _, row in df.head().iterrows():
    print(f"🔍 Disease: {row['disease']} | Symptoms: {row['combined symptom']} | Embedding: {get_symptom_embedding(row['combined symptom'], embeddings)}")


print("✅ X Shape:", X.shape)
print("✅ y Length:", len(y))


import os
print("✅ Model Exists:", os.path.exists("random_forest_model.pkl"))
print("✅ Label Encoder Exists:", os.path.exists("label_encoder.pkl"))


def get_symptom_embedding(symptoms, embeddings):
    if isinstance(symptoms, str):  # Convert string to list
        symptoms = eval(symptoms) if symptoms.startswith("[") else [symptoms]

    valid_symptoms = [symptom for symptom in symptoms if symptom in embeddings]
    if not valid_symptoms:
        return None  # If no valid symptoms, return None

    symptom_vectors = np.array([embeddings[symptom] for symptom in valid_symptoms])
    return np.mean(symptom_vectors, axis=0)  # Average symptom vectors


X = []
y = []

for _, row in df.iterrows():
    embedding = get_symptom_embedding(row["combined symptom"], embeddings)
    if embedding is not None:
        X.append(embedding)
        y.append(row["disease_encoded"])

X = np.array(X, dtype=float)  # Convert to numerical array
y = np.array(y)


print(f"✅ Sample X: {X[:3]}")  # Should print numerical values, not text
print(f"✅ Sample y: {y[:3]}")  # Should print encoded disease labels
print(f"✅ Data Shape: X={X.shape}, y={y.shape}")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

print("✅ Model training successful!")


joblib.dump(rf_model, "random_forest_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("✅ Model & Encoder saved successfully!")


# ✅ Load Model & Encoder
rf_model = joblib.load("random_forest_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ✅ Example User Symptoms
user_symptoms = ["fever", "headache", "itching"]  # Change this to test other symptoms

# ✅ Predict Disease
predicted_disease, confidence = predict_disease_rf(user_symptoms, rf_model, label_encoder, embeddings)

print(f"🩺 Predicted Disease: {predicted_disease} (Confidence: {confidence:.2f}%)")


accuracy = rf_model.score(X_test, y_test) * 100
print(f"📊 Model Accuracy: {accuracy:.2f}%")


import joblib

# ✅ Save the trained Random Forest model
joblib.dump(rf_model, "random_forest_model.pkl")

# ✅ Save the label encoder (to decode predicted disease labels)
joblib.dump(label_encoder, "label_encoder.pkl")

print("✅ Model & Encoder saved successfully!")


# ✅ Load the trained Random Forest model
rf_model = joblib.load("random_forest_model.pkl")

# ✅ Load the label encoder
label_encoder = joblib.load("label_encoder.pkl")

print("✅ Model loaded successfully!")


import os

print("Model Exists:", os.path.exists("random_forest_model.pkl"))
print("Encoder Exists:", os.path.exists("label_encoder.pkl"))


import joblib

# ✅ Load the trained Random Forest model
#rf_model = joblib.load("random_forest_model.pkl")

# ✅ Load the Label Encoder
#label_encoder = joblib.load("label_encoder.pkl")

#print("✅ Model & Encoder loaded successfully!")


loaded=joblib.load('/content/random_forest_model.pkl')


import os

print("Model Exists:", os.path.exists("random_forest_model.pkl"))
print("Encoder Exists:", os.path.exists("label_encoder.pkl"))


def predict_disease_rf(user_symptoms, model, label_encoder, embeddings):
    embedding = get_symptom_embedding(user_symptoms, embeddings)
    if embedding is None:
        print("❌ No valid symptoms found in embeddings! Cannot predict disease.")
        return None

    probabilities = model.predict_proba([embedding])[0]
    top_index = probabilities.argmax()
    predicted_disease = label_encoder.inverse_transform([top_index])[0]
    confidence = probabilities[top_index] * 100  # Convert to percentage

    print(f"🩺 Predicted Disease: {predicted_disease} (Confidence: {confidence:.2f}%)")
    return predicted_disease, confidence

# ✅ Example Usage
user_symptoms = ["fever","headache"]
predicted_disease, confidence = predict_disease_rf(user_symptoms, rf_model, label_encoder, embeddings)

exlapin this code and give the code without error and also give me the step by step process to create a model as a backend for the project 
