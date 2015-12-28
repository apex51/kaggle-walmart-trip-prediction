# Kaggle's Walmart Trip Type Prediction

## Introduction

This is code for [kaggle's walmart trip prediction][1]. The final model is a 2-level stacking ensemble in the usual fasion.

Feature Engineering:
* I treat all the fineline and department desc using bag of words for each trip. Then I generate a vector for each trip.
* The above vector I tried in three fashions: tfidf, original count, L1 regularization with no idf.
* Due to such high dimensions, I use xgb's feat importance as a selection method to get 2000~4000 features.
* I handmade several feats but I only found this feat useful: total bought num
* In the last 48h I also used upc to improve the final result: select the first one, the first two and the first three digits using bag of words method. This improved my final score from 0.551x to 0.535x.

First level:
* 2 * Multinomial Naive Bayes using tfidf and cout feats.
* 3 * Random Forest using no fineline, l1 and count feats, each with Probablistic Calibration.
* 2 * XGB using l1 and count feats.
* 2 * nn using l1 and count feats.
* 2 * kNN using count feats, with cosine and euclidean distances.

Second level:
* 1 * XGB
* 1 * nn

## Dependencies
* Python 2.7.6
    * Lasagne 0.1.dev
    * nolearn 0.5
    * numpy 1.8.2
    * pandas 0.13.1
    * scikit-learn 0.16.1
    * scipy 0.13.3
    * Theano 0.7.0
    * xgboost 0.4

[1]: https://www.kaggle.com/c/walmart-recruiting-trip-type-classification
