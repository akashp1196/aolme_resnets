import sys
import json
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

y_actual = []
y_pred = []
with open(sys.argv[1], 'r') as json_file:
    data = json.load(json_file)

    correct = 0
    total = 0
    for p in data['results']:
        trueClass = 'writing'
        if not trueClass in p:
            trueClass = 'typing'
        
        y_actual.append(trueClass)
        
        label1 = data['results'][p][0]['label']
        label2 = data['results'][p][1]['label']
        score1 = data['results'][p][0]['score']
        score2 = data['results'][p][1]['score']

        if score1 > score2:
            y_pred.append(label1)
            if label1 == trueClass:
                correct = correct + 1;
        else:
            y_pred.append(label2)
            if label2 == trueClass:
                correct = correct + 1;

        total = total + 1
    
    # Accuracy
    accuracy = (correct / total) * 100
    print('Accuracy: ' + str(accuracy) + '%')

    # Confusion Matrix
    labels = ['typing', 'writing']
    cm = confusion_matrix(y_actual, y_pred)
    df_cm = pd.DataFrame(cm, labels, labels)
    print(pd.crosstab(pd.Series(y_actual, name='Actual'), pd.Series(y_pred, name='Predicted')))
    #hm = sn.heatmap(df_cm, annot=True, cmap='Blues')
    #plt.show()


