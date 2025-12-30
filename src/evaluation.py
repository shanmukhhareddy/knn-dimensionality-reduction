from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score,f1_score

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    precision__score = precision_score(y_test,y_pred)
    recall__score=recall_score(y_test,y_pred)
    f1__score= f1_score(y_test,y_pred)

    return accuracy, cm, precision__score, recall__score ,f1__score
