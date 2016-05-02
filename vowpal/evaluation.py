import csv
from sklearn.metrics import confusion_matrix

pred_reader = csv.reader(open("predict.csv"))
y_pred=[]
y_true=[]


for i in pred_reader:
    i=[float(i) for i in i[1:]]
    j=i.index(max(i))
    if(j==3):
        y_pred.append("open")
    else:
        y_pred.append("closed")

y_true=[]
test_reader = csv.reader(open("test_split.csv"))
test_reader.next()
for line in test_reader:
    if line[14]=='open':
        y_true.append("open")
    else:
        y_true.append("closed")

print confusion_matrix(y_true , y_pred , labels=["open", "closed"])
