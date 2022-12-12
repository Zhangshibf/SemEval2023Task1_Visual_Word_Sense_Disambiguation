import pandas as pd
import spacy
import torch
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
a = pd.read_csv("/home/CE/zhangshi/SemEval23/train.data.v1.txt",sep = "\t",header=None)
keywords = list(a[0])
phrases = list(a[1])

# Load the BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Generate the word embeddings for each phrase using BERT
embeddings = []
for phrase in phrases:
    input_ids = tokenizer.encode(phrase, return_tensors="pt")
    with torch.no_grad():
        outputs = model(input_ids)
        last_hidden_states = outputs[0]
    embeddings.append(last_hidden_states[0][0].numpy())


# Use elbow method to determine optimal number of clusters
sse = []
for k in range(19, 50):
    kmeans = KMeans(n_clusters=k, random_state=1)
    kmeans.fit(embeddings)
    sse.append(kmeans.inertia_)

for a,b in enumerate(sse):
    print(a)
    print(b)
    print("----------")
# Plot the results to visualize the elbow
plt.plot(range(19, 50), sse)
plt.xlabel("Number of clusters")
plt.ylabel("SSE")
plt.show()

"""
# Use the optimal number of clusters to fit the k-means model
optimal_k = # determined using elbow method
kmeans = KMeans(n_clusters=optimal_k, random_state=1)
kmeans.fit(embeddings)

# Generate clusters for each phrase
clusters = kmeans.predict(embeddings)

# Use silhouette analysis to evaluate the quality of the clusters
silhouette_avg = silhouette_score(embeddings, clusters)
print("The average silhouette score is:", silhouette_avg)"""