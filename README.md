# Somali Homographs NLP
This repository contains the computational analysis code for the Springer-published paper "Creating and Analyzing a Dictionary-Based Lexical Resource for Somali Homograph Disambiguation" by Abdullahi Mohamed Jibril and Abdisalam Mahamed Badel (2025). The project features the first-ever Somali homograph dataset with 1,592 unique homographs extracted from the Qaamuuska Af-Soomaaliga dictionary.
Features
The analysis includes comprehensive statistical distributions across the 26-letter Somali alphabet, semantic similarity measurements using TF-IDF and sentence transformers, and machine learning clustering with evaluation metrics. The codebase generates high-resolution visualizations for publication and provides tools for homograph frequency analysis, meaning distribution studies, and semantic clustering evaluation.


``Repository Structure``

Dataset/ - Contains the Somali homographs CSV file with definitions and translations
Plots/ - High-resolution visualization outputs (600 DPI) for analysis results
somali__homographs.py - Main analysis script with all computational functions
Somali__Homographs.ipynb - Jupyter notebook version for interactive analysis

Key Analysis Components
``Statistical Analysis``

Distribution of homographs across the 26-letter Somali alphabet
Frequency analysis of meanings per homograph (average 2.5 meanings per word)
Word length distribution and most common definition terms
Identification of most ambiguous homographs by starting letter

``Semantic Analysis``

TF-IDF vectorization for semantic similarity measurement
Sentence transformer embeddings using paraphrase-MiniLM-L6-v2 model
Cosine similarity calculations between homograph definitions
Clustering evaluation with Silhouette, Calinski-Harabasz, and Davies-Bouldin indices

``Machine Learning``

K-means clustering of homograph definitions
t-SNE visualization of semantic clusters
Comparative analysis of different embedding methods
Performance evaluation across multiple clustering algorithms

Technical Requirements
python# Core dependencies
pandas >= 1.3.0
matplotlib >= 3.5.0  
seaborn >= 0.11.0
scikit-learn >= 1.0.0
sentence-transformers >= 2.0.0
numpy >= 1.21.0
Usage
python# Load and analyze the dataset
df = pd.read_csv('Dataset/somali_homographs.csv')

``Run statistical analysis``
homograph_statistics(df)
homographs_per_alphabet(df)

``Perform semantic clustering``
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embeddings = encode_sentences(model, df['Somali_definition'].tolist())
clusters = perform_clustering(embeddings, n_clusters=20)
Research Impact
This work represents the first computational analysis of Somali homographs, contributing to low-resource language NLP research. The dataset and analysis tools support future work in Somali language processing, disambiguation systems, and comparative linguistic studies.
