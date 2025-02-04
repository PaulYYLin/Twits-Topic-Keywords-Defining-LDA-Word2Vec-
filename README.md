# Twits-Topic-Keywords-Defining-LDA-Word2Vec-

## **📌 Project Overview**
This project is a **text preprocessing pipeline** designed for **Natural Language Processing (NLP)** tasks such as **topic modeling (LDA)**, **word embeddings (Word2Vec)**, and **machine learning-based text classification**. The pipeline cleans and processes raw text data to enhance model performance and efficiency.

---

## **📂 Project Structure**
```
├── Twits_Topic_Keywords_Defining_(LDA,_Word2Vec)_.ipynb      # Main script for text cleaning and processing
├── (data - urls)                      # Directory for input raw text data
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
```

---

## **🔹 Features of the Preprocessing Pipeline**
The pipeline consists of:
### text-cleaning steps:
1. **Lowercasing & Punctuation Removal**
   - Converts all text to lowercase for consistency.
   - Removes punctuation using regex (`re.sub`).

2. **Stop Word Removal**
   - Eliminates common stop words (e.g., *the, is, and*) that add little meaning.

3. **Lemmatization**
   - Uses the **WordNet Lemmatizer** to convert words to their base form.
   - Example: `running` → `run`, `better` → `good`.

4. **Extracting Nouns & Verbs Only**
   - Filters words to keep only **nouns** and **verbs**, ensuring the text remains semantically rich.

5. **Removing Numbers & Short Words**
   - Eliminates numerical values.
   - Filters out words with length ≤ 2 to remove meaningless tokens.

### 📌 parameters optimizing
Using [Optuna](https://optuna.org/) to make parameters trials to find the best parameters set.

Parameters Select:
| **Parameter**   | **Description**                           | **Range**   | **Impact**  |
|---------------|-----------------------------------|-----------|------------|
| `num_topics`  | Number of topics in LDA         | 2 - 10    | Controls topic granularity |
| `alpha`       | Document-topic distribution smoothing | 0.01 - 1.0 | Higher values distribute topics more evenly across documents |
| `passes`      | Number of training passes       | 5 - 20    | More passes improve learning but increase training time |
| `iteration`   | Iterations per pass             | 50 - 200  | More iterations improve model quality but slow down training |


#### **Objective Function Calculation**
The objective function is designed to **maximize coherence score** while **minimizing perplexity and training time**:

$
\text{objective value} = (1 - \text{coherence score}) * 0.7 + \text{normalized perplexity} * \lambda_{\text{perplexity}} + \text{normalized train time} * \lambda{\text{time}}
$

#### **Explanation of Terms:**
- **Coherence Score (`coherence_score`)** → Higher values indicate better topic quality.
- **Perplexity (`normalized_perplexity`)** → Lower values indicate better topic modeling.
- **Training Time (`normalized_train_time`)** → Lower values indicate faster training efficiency.
- **Weights:**
  - $0.7$ → Prioritizes coherence score.
  - $λ_{\text{perplexity}}$ → Controls the weight of perplexity.
  - $λ_{\text{time}}$ → Controls the weight of training time.

#### **Target:**
- **Minimize `objective_value`** to find the best hyperparameter combination.

### 📌 Topic Keywords Defining

Utilizes Word Embeddings from a Word2Vec pretrained model (trained with the glove-twitter-25 dataset).

Computes Cosine Similarity to determine the top 3 closest words to the centroid (mean) for each topic.

These closest words are chosen as representative keywords for the topic.

### 📌 Visualization

Dimensionality Reduction: Applies PCA (Principal Component Analysis) to reduce topics into 2D space with the two main principal components.

Plotting: Displays topics on a scatter plot, showing their relative positions and relationships

---

## **🚀 Installation & Setup**
### **Install Dependencies**
Ensure you have Python **3.11** installed, then install required dependencies:
```bash
pip install -r requirements.txt
```

---

## **📌 Example Input & Output**
### **Before Preprocessing:**
```
"Please, consider the following... my new VR Space Lab is available now on Amazon —
and nominated for Toy of the Year! 85 pieces set, with hands-on experiments and VR goggles to step into."
```

### **After Preprocessing:**
```
['consider', 'follow', 'space', 'year', 'piece', 'set', 'hand', 'experiment',
 'goggle', 'step', 'spaceship', 'reality']
```
### **Keywords Defining**
image.png

---

## **📌 Applications**
This preprocessing pipeline is essential for:
- **Topic Modeling** (e.g., **Latent Dirichlet Allocation - LDA**)
- **Word Embeddings** (e.g., **Word2Vec**)
- **Text Classification**
- **Natural Language Understanding (NLU)**
- **Machine Learning & Deep Learning Models**

