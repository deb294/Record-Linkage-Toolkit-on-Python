import numpy as np
import recordlinkage as rl
from recordlinkage.datasets import load_krebsregister

krebs_data, krebs_match = load_krebsregister(missing_values=0)
print(krebs_data)

des = krebs_data.describe().T
print(des)

golden_pairs = krebs_data[0:5000]
golden_matches_index = golden_pairs.index & krebs_match # 2093 matching pairs
#print(golden_matches_index)
#np.savetxt("golden_matches_index.csv",golden_matches_index)

# Initialize the classifier
logreg = rl.LogisticRegressionClassifier()

# Train the classifier
logreg.learn(golden_pairs, golden_matches_index)
print ("Intercept: ", logreg.intercept)
print ("Coefficients: ", logreg.coefficients)

# Predict the match status for all record pairs
result_logreg = logreg.predict(krebs_data)

print(len(result_logreg))

conf_logreg = rl.confusion_matrix(krebs_match, result_logreg, len(krebs_data))
print(conf_logreg)

# The F-score for this prediction is
print(rl.fscore(conf_logreg))

intercept = -9
coefficients = [2.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0]

logreg = rl.LogisticRegressionClassifier(coefficients, intercept)

# predict without calling LogisticRegressionClassifier.learn
matches = logreg.predict(krebs_data)
print (len(matches))

conf_logreg = rl.confusion_matrix(krebs_match, matches, len(krebs_data))
print(conf_logreg)

# The F-score for this classification is
print(rl.fscore(conf_logreg))

# Train the classifier
nb = rl.NaiveBayesClassifier()
nb.learn(golden_pairs, golden_matches_index)

# Predict the match status for all record pairs
result_nb = nb.predict(krebs_data)

print(len(result_nb))

conf_nb = rl.confusion_matrix(krebs_match, result_nb, len(krebs_data))
print(conf_nb)

# The F-score for this classification is
print(rl.fscore(conf_nb))

# Train the classifier
svm = rl.SVMClassifier()
svm.learn(golden_pairs, golden_matches_index)

# Predict the match status for all record pairs
result_svm = svm.predict(krebs_data)

print(len(result_svm))

conf_svm = rl.confusion_matrix(krebs_match, result_svm, len(krebs_data))
print(conf_svm)

# The F-score for this classification is
print(rl.fscore(conf_svm))

kmeans = rl.KMeansClassifier()
result_kmeans = kmeans.learn(krebs_data)

# The predicted number of matches
print(len(result_kmeans))


cm_kmeans = rl.confusion_matrix(krebs_match, result_kmeans, len(krebs_data))

print(rl.fscore(cm_kmeans))

# Train the classifier
ecm = rl.ECMClassifier()
result_ecm = ecm.learn((krebs_data > 0.8).astype(int))

print(len(result_ecm))

conf_ecm = rl.confusion_matrix(krebs_match, result_ecm, len(krebs_data))
print(conf_ecm)

# The F-score for this classification is
print(rl.fscore(conf_ecm))
