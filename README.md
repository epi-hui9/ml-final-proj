# Classification of Philosophical Schools Using Text-Based Models

The central goal of this project is to investigate whether different schools of philosophical thought possess distinct linguistic patterns that can be identified and classified by machine learning algorithms. We aim to answer the question: Can a model accurately predict the philosophical school (e.g., Rationalism, Empiricism, Stoicism) to which a sentence belongs, based solely on its text?

## Usage Instructions

## Data Source

This project uses the [History of Philosophy](https://www.kaggle.com/datasets/kouroshalizadeh/history-of-philosophy) dataset from Kaggle as the primary data source. Please download the dataset and place it in the `data/` directory before running the code.

## Models Overview

| Model | Strategy | How it Works |
| :--- | :--- | :--- |
| Decision Tree | Native | The tree is built to separate all 13 classes. Each leaf node is assigned a class. |
| Random Forest | Native | Averages the votes from many native multi-class Decision Trees. |
| Linear SVM (LinearSVC) | One-vs-Rest (OvR) | Trains 13 independent binary models (one for each class vs. the rest). |
| Logistic Regression | Multinomial (Softmax) | Trains a single model that outputs a probability for each of the 13 classes at once. |

## 01 Data Preprocessing

From the EDA of the original full dataset, we learned that:

- No missing values exist in the dataset.
- The classes are not evenly distributed. "Analytic" has 55k entries, while "Stoicism" only has 2.5k. Our models will be much better at predicting the common classes. When we look at the results, we must check the F1-score for "Stoicism" to see if the model is just ignoring it.
- The average sentence is 151 characters, but the max is 2,649. This is a wide range, but TF-IDF is generally robust to this.

## 02 Base Modeling

This very first step involved basic model training using TF-IDF vectorization and two classifiers: Decision Tree and Random Forest. This served as a baseline for more complex models.

1. Decision Tree (Accuracy: 36.8%):

    - This is our low-end baseline. With 13 classes, random guessing would be ~7.7% (1/13), so 37% is significantly better than chance.
    - However, the F1-scores are low across the board (e.g., stoicism: 0.10), which confirms this model is struggling.

2. Random Forest (Accuracy: 50.6%):

    - We got a ~14% jump in accuracy by moving from one tree to 100 trees. It seems that a single tree is overfitting, while the ensemble generalizes better.
    - We have imbalanced classes. 'stoicism' has a precision of 0.60, but a recall of 0.09. This means that when the model predicts 'stoicism', it's usually right, but it misses most of the actual 'stoicism' sentences.
    - From the feature importance results, the top feature is 'pron' (0.028408). This is an artifact from the lemmatizer — it's the token for "pronoun". It is noise and not a real word. We need to remove it.
    - Other top features are exactly what we'd hope to see: 'woman' (likely from "feminism"), 'god', 'man', 'idea', 'madness' (likely "nietzsche" or "continental"). This proves our core concept is working.

## 03 Model Comparison

We improved our TF-IDF vectorization by adding 'pron' to the stop words list. We trained four models: Decision Tree, Random Forest, Logistic Regression, and SVM based on the improved data.

So far, our results are:

| Model | Overall Accuracy | Weighted F1-Score | Training Time |
| :--- | :--- | :--- | :--- |
| Decision Tree | 39.2% | 0.39 | 5.27s |
| Random Forest | 50.6% | 0.50 | 3.18s |
| Logistic Regression | 63.9% | 0.64 | 0.30s |
| Linear SVM | 64.4% | 0.64 | 0.74s |

The Linear SVM is our best performing model, with Logistic Regression as a very close second. Linear models dramatically outperformed tree-based models. A possible reason is that the TF-IDF matrix is very wide (5,000 features) and sparse, which is a setting where linear models excel. Linear models are not only more accurate but also much faster to train.

Even though the performance improves with linear models, the F1-scores for minority classes like "Stoicism" remain low (0.21 for SVM). This indicates that while overall accuracy is good, the model still struggles with less represented classes.

### Results on Full Dataset

After verifying model behavior on the small sample, we reran all experiments on the **full dataset**.  
This produces much more stable and reliable results:

| Model               | Overall Accuracy | Weighted F1-Score | Training Time |
| :------------------ | :--------------: | :----------------: | :-----------: |
| Decision Tree       | 46.6%            | 0.46               | 63.10s        |
| Random Forest       | 58.7%            | 0.58               | 120.33s       |
| Logistic Regression | 69.6%            | 0.69               | 3.07s         |
| Linear SVM          | 69.5%            | 0.69               | 12.08s        |

As expected, using all **360k samples** significantly improves performance across all models —  
especially for minority classes like **Stoicism**, whose F1-score more than doubled on the full dataset.


### Sample vs. Full Dataset Comparison

To understand how much we benefit from training on all 360k examples, we compare the metrics from the **small stratified sample** (used in early experiments) and the **full dataset** side by side:

| Model               | Accuracy (Sample) | Accuracy (Full) | Δ Accuracy | Weighted F1 (Sample) | Weighted F1 (Full) | Δ F1  |
| :------------------ | :---------------: | :-------------: | :--------: | :-------------------: | :----------------: | :---: |
| Decision Tree       | 39.2%             | 46.6%           | +7.4 pts   | 0.39                  | 0.46               | +0.07 |
| Random Forest       | 50.6%             | 58.7%           | +8.1 pts   | 0.50                  | 0.58               | +0.08 |
| Logistic Regression | 63.9%             | 69.6%           | +5.7 pts   | 0.64                  | 0.69               | +0.05 |
| Linear SVM          | 64.4%             | 69.5%           | +5.1 pts   | 0.64                  | 0.69               | +0.05 |

Key observations:

- **All models improve** by roughly **5–8 percentage points** in accuracy when moving from the sample to the full dataset.
- The **ranking of models does not change**: linear models (Logistic Regression, Linear SVM) remain clearly stronger than tree-based models, which confirms that our early small-sample experiments were already pointing in the right direction.
- Tree-based models gain the most in absolute terms (+7–8 pts), but they are still clearly behind the linear models on this high-dimensional, sparse TF-IDF representation.
- On the full dataset, minority classes such as **Stoicism** see a large boost in F1-score (e.g., Linear SVM Stoicism F1 from ~0.21 on the sample to **0.54** on the full data), showing that using the entire corpus is crucial for rare schools.


## 04 SVM Analysis


After identifying Linear SVM as the best performing model, we conducted deeper analysis to examine how the model makes decisions and which linguistic features are most informative for each philosophical school.

### Top 10 Indicative Features for Each School

The following are the top positively weighted TF-IDF tokens for each of the 13 classes. These tokens represent the strongest textual signals that the SVM relies on during classification:

[ANALYTIC]: nozick, counterfactual, frege, carnap, quine, semantical, kaplan, nixon, unicorn, objct  
[ARISTOTLE]: incontinent, enthymeme, semen, nutriment, excellence, incontinence, empedocles, reputable, concoction, grub  
[CAPITALISM]: liquidity, bounty, tax, unemployment, investment, butcher, saving, clergy, scarcity, employment  
[COMMUNISM]: kautsky, imperialism, capitalistic, workpeople, marxism, engels, factory, inspector, bourgeois, colonial  
[CONTINENTAL]: levinas, artaud, confinement, madness, pinel, oedipal, bic, foucault, derrida, familial  
[EMPIRICISM]: betwixt, coue, encrease, uneasiness, conformable, commonwealth, solidity, innate, contrivance, immaterial  
[FEMINISM]: housework, housewife, woman, racism, lynching, mme, rape, motherhood, racist, femininity  
[GERMAN_IDEALISM]: determinateness, purposiveness, cognition, sache, purposive, alaska, supersensible, sublate, sensuous, principien  
[NIETZSCHE]: zarathustra, verily, wagner, ye, christianity, germans, instinct, yea, spake, priest  
[PHENOMENOLOGY]: givenness, ontically, dasein, pregiven, epoche, unconcealment, factical, tactile, ontologically, factically  
[PLATO]: expertise, dion, critias, clinias, glaucon, cratylus, phaedrus, theaetetus, euthydemus, cebes  
[RATIONALISM]: bayle, lens, enor, vortex, lhe, jupiter, fiber, wherefore, prop, def  
[STOICISM]: whatsoever, doth, conceit, unto, hurt, discretion, commend, worldly, thou, meat  

These keywords validate our assumption that different philosophical schools exhibit distinctive linguistic patterns.  
For example:

- *Nietzsche* includes highly idiosyncratic archaic terms (“verily”, “ye”, “spake”).  
- *Feminism* contains gender- and society-related vocabulary (“housewife”, “racism”, “rape”).  
- *Phenomenology* and *Continental* are rich in Heideggerian/German terminology (“dasein”, “givenness”, “facticity”).  
- *Stoicism* shows early Christian/archaic English structures (“thou”, “doth”).  

## 05 Discussion


### Model Performance Insights

Linear models (SVM, Logistic Regression) significantly outperform tree-based models on high-dimensional sparse TF-IDF features.

Tree models overfit easily due to the extremely wide feature space (5,000+ tokens).

Full-dataset training greatly improves minority-class performance, especially “Stoicism,” whose F1-score more than doubled.

### Error Patterns

Misclassification mostly occurs among conceptually similar schools (e.g., Phenomenology vs Continental, Rationalism vs Empiricism).

Schools with limited data (e.g., Stoicism) show unstable predictions when trained on small samples, but improve significantly on the full dataset.

### Linguistic Distinctiveness

The SVM analysis reveals clear linguistic clusters—for instance Nietzsche uses highly archaic spellings, Feminism centers on gender-political terminology, and Phenomenology heavily uses German philosophical terms.

## 06 future work

There are several directions that could further improve our results:

Use more advanced text representations such as sentence embeddings or transformer-based models (e.g., BERT), which may capture deeper semantic patterns than TF-IDF.

Address class imbalance, especially for rare schools like Stoicism, using class weighting or oversampling techniques.

Perform deeper error analysis to understand why certain schools (e.g., Continental vs. Phenomenology) are frequently confused and refine preprocessing accordingly.

Try additional models, including neural networks or regularized linear models with different hyperparameters, to see whether performance can be pushed beyond the current ~70% accuracy.

## 07 Github Repository

https://github.com/epi-hui9/ml-final-proj.git
