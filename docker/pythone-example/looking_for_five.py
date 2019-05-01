from sklearn.datasets import fetch_mldata
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

#downloading data
mnist = fetch_mldata('MNIST original')
X, y = mnist["data"], mnist["target"]
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

#shuffling data
shuffled_indices = np.random.permutation(60000)
X_train, y_train = X_train[shuffled_indices], y_train[shuffled_indices]

#binary classifier
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
sgd_classifier = SGDClassifier()
sgd_classifier.fit(X_train, y_train_5)
#if sgd_classifier.predict([X[36000]])[0]:
#	print("Good prediction!!!")

#model evaluating
result = cross_val_score(sgd_classifier, X_train, y_train_5, cv=3, scoring="accuracy")
#print(result)

class Never5Classifier(BaseEstimator):
	def fit(self, X, y=None):
		pass
	def predict(self, X):
		return np.zeros((len(X), 1), dtype=bool)

never5Classifier = Never5Classifier()
result = cross_val_score(never5Classifier, X_train, y_train_5, cv=3, scoring="accuracy")
#print(result)


#accuracy is not the best estimator!
#confusion matrix
y_train_pred = cross_val_predict(sgd_classifier, X_train, y_train_5, cv=3)
c_matrix = confusion_matrix(y_train_5, y_train_pred)
print(c_matrix)

#precision and recall (pelnosc)

precision = precision_score(y_train_5, y_train_pred)
recall = recall_score(y_train_5, y_train_pred)
print("precision:", precision)
print("recall", recall)

#f1 score (harmonic mean)
f1 = f1_score(y_train_5, y_train_pred)
print("f1:", f1)

#threshold - balancing between precision and recall

#one digit
#print("Target:", y[20])
y_score = sgd_classifier.decision_function([X[20]])
#print("y_score", y_score)
threshold = 0
y_prediction = (y_score > threshold)
#print("prediction: ", str(y_prediction), " with treshold: " + str(threshold))

#all digits
y_scores = cross_val_predict(sgd_classifier, X_train, y_train_5, cv=3, method="decision_function")
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

# [:-1] means take all from 0 to penultimate element. [1,2,3] -> [1,2]
#plt.plot(thresholds, precisions[:-1], "b--", label="Precyzja")
#plt.plot(thresholds, recalls[:-1], "g-", label="Pełność")
#plt.xlabel("Próg")
#plt.legend(loc="center left")
#plt.ylim([0,1])
#plt.show()

#threshold = 70000
y_train_pred_90 = (y_scores > 70000)

print(precision_score(y_train_5, y_train_pred_90))
print(recall_score(y_train_5, y_train_pred_90))

#ROC Curve (receiver operating characteristic)
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

#plt.plot(fpr, tpr, linewidth=2, label=None)
#plt.plot([0,1],[0,1], "k--")
#plt.axis([0,1,0,1])
#plt.xlabel("Odsetek fałszywie pozytywnych")
#plt.ylabel("Odsetek prawdziwie pozytywnych")
#plt.show()

#AUC area undet the curve, the closer to 1 the better
auc = roc_auc_score(y_train_5, y_scores)
print("AUC score:", auc)

#Learning RandomForestClassifier

forest_classifier = RandomForestClassifier()
y_probas_forest = cross_val_predict(forest_classifier, X_train, y_train_5, cv=3, method="predict_proba")

#converting data
y_scores_forest = y_probas_forest[:, 1]
fpr_forest, tpr_forest, threshold_forest = roc_curve(y_train, y_scores,y_scores_forest)

#then making plot and comparing with SGDClassifier
#It seems that it is a better classifier (if you plot it :))
#checking precision and recall

forest_result = roc_auc_score(y_train_5, y_scores_forest)
print("forest AUC score", forest_result)

