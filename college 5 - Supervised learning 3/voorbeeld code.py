from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier

X_train_full = train_df.drop("Survived", axis=1)
Y_train_full = train_df["Survived"]
X_test_full  = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


X_train, X_test, y_train, y_test = train_test_split(X_train_full, Y_train_full, random_state=43, test_size=0.10)

tuned_parameters = [{'kernel': ['rbf'], 'gamma': np.logspace(-9, 3, 13),
                     'C': np.logspace(-6, -1, 10)},
                    {'kernel': ['linear'], 'C': np.logspace(-6, -1, 10)},
                    {'kernel': ['poly'], 'gamma': np.logspace(-9, 3, 13), 'C': np.logspace(-6, -1, 10)},
                    {'kernel': ['sigmoid'], 'gamma': np.logspace(-9, 3, 13), 'C': np.logspace(-6, -1, 10)}]

scoringArray = ['balanced_accuracy', "average_precision", "f1", "accuracy", "f1_micro", "f1_weighted"]

maxPredictScore = 0
bestScoreParam = dict()
bestEstimator = dict()
resultArray = []

print (X_train.shape, y_train.shape)
for score in scoringArray:
    print ("Scoring param: {}".format(score))
    bagging = BaggingClassifier(GridSearchCV(SVC(), tuned_parameters, cv=None, scoring=score, iid=False),
                                oob_score=True, warm_start=False, random_state=42, max_samples=0.9, max_features=0.5)
     
    cv_results = cross_validate(bagging, X_train, y_train, cv=20, return_train_score=True, return_estimator=True, n_jobs=-1)
    svMax = cv_results['estimator'][np.argmax(cv_results['test_score'])]                           

    predictScore = round(svMax.score(X_test, y_test) * 100, 2)
    if (maxPredictScore < predictScore):
        maxPredictScore = predictScore
        bestScoreParam = score
        bestEstimator = svMax
    print (predictScore)
    resultArray.append({"score":score, "cvResults" : cv_results, "svMax": svMax, "maxScore":predictScore})

print ("Best score: {} with scoring: {}".format(bestScoreParam, maxPredictScore))