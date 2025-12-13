# **Classification of Philosophical Schools Using Text-Based Models**

---

## **1\. Introduction**

The central goal of this project is to investigate whether different schools of philosophical thought possess distinct linguistic patterns that can be identified and classified by machine learning algorithms. We aim to answer the question: Can a model accurately predict the philosophical school (e.g., Rationalism, Empiricism, Stoicism) to which a sentence belongs, based solely on its text?

---

## **2\. Dataset**

### **2.1 Data Source**

This project uses the [History of Philosophy](https://www.kaggle.com/datasets/kouroshalizadeh/history-of-philosophy) dataset from Kaggle as the primary data source. Please download the dataset and place it in the `data/` directory before running the code.

---

### **2.2 Dataset Properties**

From the EDA of the original full dataset, we learned that:

* No missing values exist in the dataset.  
* The classes are not evenly distributed. "Analytic" has 55k entries, while "Stoicism" only has 2.5k. Our models will be much better at predicting the common classes. When we look at the results, we must check the F1-score for "Stoicism" to see if the model is just ignoring it.  
* The average sentence is 151 characters, but the max is 2,649. This is a wide range, but TF-IDF is generally robust to this.

---

## **3\. Methods Overview**

### **3.1 Text Representation**

All models in this project rely on TF-IDF (Term Frequency–Inverse Document Frequency) representations of lemmatized sentences.

TF-IDF is well-suited for philosophical texts because it emphasizes distinctive, school-specific terminology while down-weighting common words.

---

### **3.2 Models Overview**

| Model | Strategy | How it Works |
| ----- | ----- | ----- |
| Decision Tree | Native | The tree is built to separate all 13 classes. Each leaf node is assigned a class. |
| Random Forest | Native | Averages the votes from many native multi-class Decision Trees. |
| Linear SVM (LinearSVC) | One-vs-Rest (OvR) | Trains 13 independent binary models (one for each class vs. the rest). |
| Logistic Regression | Multinomial (Softmax) | Trains a single model that outputs a probability for each of the 13 classes at once. |
| **TF-IDF Neural Network** | Feed-forward NN | Learns non-linear interactions between TF-IDF features via dense layers. |

---

## **4\. Base Modeling (Tree-Based Baselines)**

This very first step involved basic model training using TF-IDF vectorization and two classifiers: Decision Tree and Random Forest. This served as a baseline for more complex models.

### **4.1 Decision Tree**

**Accuracy: 36.8%**

* This is our low-end baseline. With 13 classes, random guessing would be \~7.7% (1/13), so 37% is significantly better than chance.  
* However, the F1-scores are low across the board (e.g., stoicism: 0.10), which confirms this model is struggling.

---

### **4.2 Random Forest**

**Accuracy: 50.6%**

* We got a \~14% jump in accuracy by moving from one tree to 100 trees. It seems that a single tree is overfitting, while the ensemble generalizes better.  
* We have imbalanced classes. 'stoicism' has a precision of 0.60, but a recall of 0.09. This means that when the model predicts 'stoicism', it's usually right, but it misses most of the actual 'stoicism' sentences.  
* From the feature importance results, the top feature is 'pron' (0.028408). This is an artifact from the lemmatizer — it's the token for "pronoun". It is noise and not a real word. We need to remove it.  
* Other top features are exactly what we'd hope to see: 'woman' (likely from "feminism"), 'god', 'man', 'idea', 'madness' (likely "nietzsche" or "continental"). This proves our core concept is working.

---

## **5\. Linear Model Comparison**

We improved our TF-IDF vectorization by adding 'pron' to the stop words list. We trained four models: Decision Tree, Random Forest, Logistic Regression, and SVM based on the improved data.

### **5.1 Results on Stratified Sample**

| Model | Overall Accuracy | Weighted F1-Score | Training Time |
| ----- | ----- | ----- | ----- |
| Decision Tree | 39.2% | 0.39 | 5.27s |
| Random Forest | 50.6% | 0.50 | 3.18s |
| Logistic Regression | 63.9% | 0.64 | 0.30s |
| Linear SVM | 64.4% | 0.64 | 0.74s |

The Linear SVM is our best-performing model, with Logistic Regression as a very close second.

---

### **5.2 Results on Full Dataset**

After verifying model behavior on the small sample, we reran all experiments on the **full dataset**.

| Model | Overall Accuracy | Weighted F1-Score | Training Time |
| ----- | ----- | ----- | ----- |
| Decision Tree | 46.6% | 0.46 | 63.10s |
| Random Forest | 58.7% | 0.58 | 120.33s |
| Logistic Regression | 69.6% | 0.69 | 3.07s |
| Linear SVM | 69.5% | 0.69 | 12.08s |

---

## **6\. TF-IDF Neural Network**

### **6.1 Model Architecture**

To explore whether non-linear models can outperform linear classifiers on TF-IDF features, we implemented a feed-forward neural network:

* Input: 5,000-dimensional TF-IDF vector  
* Dense (256 units) \+ ReLU  
* Dropout  
* Dense (128 units) \+ ReLU  
* Dropout  
* Output layer (13 classes, softmax)

The model is trained using cross-entropy loss and evaluated with accuracy and per-class F1-score.

---

### **6.2 Neural Network Results (Full Dataset)**

* **Overall Accuracy:** **70.7%**  
* **Weighted F1-score:** **0.71**  
* Best-performing model in the project so far.

Notably, the neural network improves performance on minority classes:

* **Stoicism F1-score:** **0.58**, higher than all linear baselines.

---

### **6.3 Comparison with Linear Models**

Compared to Linear SVM (69.5%) and Logistic Regression (69.6%), the TF-IDF neural network achieves a modest but consistent improvement.

This suggests that, with sufficient data (360k samples), a simple feed-forward neural network can learn non-linear interactions between TF-IDF features that linear models cannot capture.

---

## **7 Embedding-based Neural Models**

### **7.1 Embedding Feed-Forward Neural Network**

We used spaCy’s \`en\_core\_web\_md\` model to map words to 300-dimensional dense vectors (GloVe). For each sentence, we averaged all word vectors to create a single 300-dimensional input vector.

We achieved the following result with the full 360808-row dataset:

* Overall Accuracy: 51.0%  
* Training Time: \~56 minutes (3,364s)

This model significantly underperformed compared to our linear baselines (51% vs \~70%). The poor performance confirms that by "averaging" the word vectors, we destroyed the specific signals (keywords) necessary for classification. It smoothed out the "linguistic fingerprints" (like specific jargon or proper nouns) that we identified as crucial in the SVM analysis.

---

### **7.2 LSTM Sequence Model**

To address the "averaging" problem, we implemented a sequence-based model that processes words one by one, preserving both semantic meaning and sentence structure. 

Method:

* Vocabulary: We built a custom vocabulary of 88,623 unique words from the full dataset.   
* Architecture: An LSTM (Long Short-Term Memory) network with:   
  * Embedding Layer (initialized with pre-trained GloVe weights)  
  * LSTM Layer (128 hidden units, 2 stacked layers, dropout=0.5)  
  * Fully Connected Output Layer

Results (Full Dataset):

* Overall Accuracy: 75.0%  
* Training Time: \~63 minutes (3,780s)

This model is our **BEST** model in the project. It outperformed all other models, including the Linear SVM (69.5%) and the TF-IDF Neural Network (70.7%). 

---

### **7.3 Results and Comparison**

The LSTM's success (75%) vs. the FFNN's failure (51%) proves that sequence matters. While TF-IDF models are excellent at catching keywords, the LSTM was able to combine the power of specific vocabulary (via a massive 88k-word vocabulary) with the ability to understand the context in which those words appear. It essentially learned to "read" the philosophy rather than just "spot keywords." 

---

## **8\. SVM Feature Analysis**

### **Top 10 Indicative Features for Each School**

\[ANALYTIC\]: nozick, counterfactual, frege, carnap, quine, semantical, kaplan, nixon, unicorn, objct

\[ARISTOTLE\]: incontinent, enthymeme, semen, nutriment, excellence, incontinence, empedocles, reputable, concoction, grub

\[CAPITALISM\]: liquidity, bounty, tax, unemployment, investment, butcher, saving, clergy, scarcity, employment

\[COMMUNISM\]: kautsky, imperialism, capitalistic, workpeople, marxism, engels, factory, inspector, bourgeois, colonial

\[CONTINENTAL\]: levinas, artaud, confinement, madness, pinel, oedipal, bic, foucault, derrida, familial

\[EMPIRICISM\]: betwixt, coue, encrease, uneasiness, conformable, commonwealth, solidity, innate, contrivance, immaterial

\[FEMINISM\]: housework, housewife, woman, racism, lynching, mme, rape, motherhood, racist, femininity

\[GERMAN\_IDEALISM\]: determinateness, purposiveness, cognition, sache, purposive, alaska, supersensible, sublate, sensuous, principien

\[NIETZSCHE\]: zarathustra, verily, wagner, ye, christianity, germans, instinct, yea, spake, priest

\[PHENOMENOLOGY\]: givenness, ontically, dasein, pregiven, epoche, unconcealment, factical, tactile, ontologically, factically

\[PLATO\]: expertise, dion, critias, clinias, glaucon, cratylus, phaedrus, theaetetus, euthydemus, cebes

\[RATIONALISM\]: bayle, lens, enor, vortex, lhe, jupiter, fiber, wherefore, prop, def

\[STOICISM\]: whatsoever, doth, conceit, unto, hurt, discretion, commend, worldly, thou, meat

---

## **9\. Discussion**

| Model | Overall Accuracy | Weighted F1-Score | Training Time |
| ----- | ----- | ----- | ----- |
| Decision Tree | 46.6% | 0.46 | 63.10s |
| Feed-Forward NN (Embeddings) | 51.0% | 0.50 | 3364.46s |
| Random Forest | 58.7% | 0.58 | 120.33s |
| Linear SVM | 69.5% | 0.69 | 12.08s |
| Logistic Regression | 69.6% | 0.69 | 3.07s |
| TFIDF NN | 70.7% | 0.71 | 274s |
| LSTM (Embeddings) | 75.0% | 0.75 | 3779.65s |

### **9.1 Model Performance Insights**

Linear models (SVM, Logistic Regression) significantly outperform tree-based models on high-dimensional sparse TF-IDF features.

Tree models overfit easily due to the extremely wide feature space (5,000+ tokens).

Full-dataset training was critical. It not only improved general accuracy but drastically helped minority classes. The "Stoicism" F1-score improved from \~0.10 in early prototypes to \~0.53 in the final LSTM model.

Linear models (SVM, Logistic Regression) significantly outperform "averaged" embedding models. The FFNN with averaged embeddings failed (51%) because it lost the keyword signals that linear models capture easily.

Most importantly, the LSTM (75%) outperformed the Linear SVM (69.5%). This refutes our earlier hypothesis that keywords alone are sufficient. While specific jargon is important, modeling the sequential structure of the text allows for higher accuracy.

---

### **9.2 Error Patterns**

Misclassification mostly occurs among conceptually similar schools (e.g., Phenomenology vs Continental, Rationalism vs Empiricism).

Schools with limited data (e.g., Stoicism) show unstable predictions when trained on small samples, but improve significantly on the full dataset.

---

## **10\. Future Work**

There are several directions that could further improve our results:

Use more advanced text representations such as transformer-based models (e.g., BERT), which may capture deeper semantic patterns than TF-IDF.

Address class imbalance, especially for rare schools like Stoicism, using class weighting or oversampling techniques.

Perform deeper error analysis to understand why certain schools (e.g., Continental vs. Phenomenology) are frequently confused and refine preprocessing accordingly.

Since TF-IDF captures keywords and LSTMs capture context, we may attempt a hybrid model that concatenates TF-IDF features with the LSTM's dense layer to get the benefits of both see whether performance can be pushed beyond the current \~70% accuracy.

---

## **11\. GitHub Repository**

[https://github.com/epi-hui9/ml-final-proj.git](https://github.com/epi-hui9/ml-final-proj.git)

