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




