import competition_utilities as cu
import features
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from threading import Thread
import numpy as np
from sklearn.metrics import confusion_matrix
import csv

train_file = "train-sample_October_9_2012_v2.csv"
feature_file1="feature_set.csv"
feature_file2="feature_set1.csv"
full_train_file = "train.csv"
test_file = "test1.csv"
submission_file = "basic_benchmark.csv"


def main():
    print("Reading the data")
    data = cu.get_dataframe(train_file)

    #print("Extracting features")
    #features.compute_features(train_file,feature_file1)

    print("Training the model")
    fea = cu.get_dataframe(feature_file1)
    rf = RandomForestClassifier(n_estimators=100, verbose=2, n_jobs=-1)
    rf.fit(fea, data["OpenStatus"][:178351])
    important_features = []
    for x, i in enumerate(rf.feature_importances_):
        if i > np.average(rf.feature_importances_):
            important_features.append([str(x),i])
    print 'Most important features:',important_features

    print("Reading test file and making predictions")
    #features.compute_features("test1.csv",feature_file2)
    test_fea = cu.get_dataframe(feature_file2)
    probs = rf.predict_proba(test_fea)

    print("Calculating priors and updating posteriors")
    new_priors = cu.get_priors(full_train_file)
    old_priors = cu.get_priors(train_file)
    probs = cu.cap_and_update_priors(old_priors,probs, new_priors, 0.01)

    y_pred=[]
    for i in probs:
        i=[float(k) for k in i]
        j=i.index(max(i))
        if(j==3):
            y_pred.append("open")
        else:
            print "hi"
            y_pred.append("closed")

    y_true=[]
    a=0
    b=0
    test_reader = csv.reader(open(test_file))
    headers=test_reader.next()
    for line in test_reader:
        if line[14]=='open':
            y_true.append("open")
            a=a+1
        else:
            y_true.append("closed")
            b=b+1
    print a
    print b

    print confusion_matrix(y_true[1:] , y_pred , labels=["open", "closed"])





    print("Saving submission to %s" % submission_file)

    cu.write_submission(submission_file, probs)

if __name__=="__main__":
    main()



