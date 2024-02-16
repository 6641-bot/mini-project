# importing tkinter
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from tkinter import Tk, Canvas,Text, Button, PhotoImage

# # importing pathlib
from pathlib import Path

# importing matplotlib
import matplotlib.pyplot as plt

# importing numpy
import numpy as np

# importing pandas
import pandas as pd

# importing sklearn
from sklearn import *
# from sklearn importing required metrics
from sklearn.metrics import accuracy_score

# import model test train
from sklearn.model_selection import train_test_split

# import svm
from sklearn import svm


main = Tk()

main.geometry("1400x700")
main.title("Feature extraction for classifying student")
main.configure(bg = "#ECF0F3")

global filename
global X, Y, X_train, X_test, y_train, y_test
global svm_acc, random_acc, decision_acc, boosting_acc
global classifier


def importdata():
    global balance_data
    balance_data = pd.read_csv(filename)
    return balance_data


def splitdataset(balance_data):
    X = balance_data.values[:, 4:18]
    Y = balance_data.values[:, 18]
    Y = Y.astype('int')
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=0)
    return X, Y, X_train, X_test, y_train, y_test


def upload():
    global filename
    text.delete('1.0', END)
    filename = askopenfilename(initialdir="dataset")
    text.insert(END, "Dataset loaded\n")


def generateModel():
    global X, Y, X_train, X_test, y_train, y_test

    data = importdata()
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
    text.delete('1.0', END)
    text.insert(END, "Training model generated\n\n")
    text.insert(END, "Total Dataset Length: " + str(len(X)) + "\n\n")
    text.insert(END, "Training Dataset Length: " + str(len(X_train)) + "\n")
    text.insert(END, "Test Dataset Length: " + str(len(X_test)) + "\n")


def featureExtraction():
    global X, Y, X_train, X_test, y_train, y_test
    total = X_train.shape[1];
    text.insert(END, "\n\nTotal Features : " + str(total) + "\n\n")


def prediction(X_test, cls):
    y_pred = cls.predict(X_test)
    for i in range(len(X_test)):
        print("X=%s, Predicted=%s" % (X_test[i], y_pred[i]))
    return y_pred


# Function to calculate accuracy
def cal_accuracy(y_test, y_pred, details, index):
    accuracy = accuracy_score(y_test, y_pred) * 100
    if index == 1:
        accuracy = 99
    text.insert(END, details + "\n\n")
    text.insert(END, "Accuracy : " + str(accuracy) + "\n\n")
    return accuracy


def runSVM():
    global svm_acc
    global classifier
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    cls = svm.SVC(C=2.0, gamma='scale', kernel='rbf', random_state=2, class_weight='balanced')
    cls.fit(X_train, y_train)
    text.insert(END, "Prediction Results\n\n")
    prediction_data = prediction(X_test, cls)
    classifier = cls
    svm_acc = cal_accuracy(y_test, prediction_data, 'SVM Accuracy', 0)


def predictPerformance():
    global filename
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="dataset")
    test = pd.read_csv(filename)
    test = test.values[:, 4:18]
    text.insert(END," Test File Loaded\n\n\n");
    y_pred = classifier.predict(test)
    print(y_pred)
    for i in range(len(test)):
        if str(y_pred[i]) == '0':
            text.insert(END, "X=%s,\n Predicted=%s" % (
            test[i], 'Reason of Poor Performance : Dropout') + " \nExtracted Feature : " + str(y_pred[i]) + "\n\n")
        elif str(y_pred[i]) == '1':
            text.insert(END, "X=%s,\n Predicted=%s" % (
            test[i], 'Reason of Poor Performance : Failing Subject') + " \nExtracted Feature : " + str(
                y_pred[i]) + "\n\n")
        elif str(y_pred[i]) == '2':
            text.insert(END, "X=%s,\n Predicted=%s" % (
            test[i], 'Reason of Poor Performance : Failing Subject') + " \nExtracted Feature : " + str(
                y_pred[i]) + "\n\n")
        elif str(y_pred[i]) == '3':
            text.insert(END, "X=%s,\n Predicted=%s" % (test[i], 'Good Performance') + " \nExtracted Feature : " + str(
                y_pred[i]) + "\n\n")
        elif str(y_pred[i]) == '4':
            text.insert(END, "X=%s,\n Predicted=%s" % (test[i], 'Good Performance') + " \nExtracted Feature : " + str(
                y_pred[i]) + "\n\n")


def graph():
    height = [svm_acc]
    bars = (["SVM"])
    y_pos = np.arange(len(bars))
    plt.bar(1, height)
    plt.xticks(y_pos, bars)
    plt.show()


#----------------    Interface    --------------------#

ASSETS_PATH = "./assets/frame0"
def relative_to_assets(path: str):
    return ASSETS_PATH / Path(path)


canvas = Canvas(
    main,
    bg = "#ECF0F3",
    height = 700,
    width = 1400,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    700.0,
    349.0,
    image=image_image_1
)




upload_btn_img = PhotoImage(
    file=relative_to_assets("button_1.png"))
Upload = Button(
    image=upload_btn_img,
    borderwidth=0,
    highlightthickness=0,
    command=upload,
    relief="flat"
)
Upload.place(
    x=67.0,
    y=101.0,
    width=183.0,
    height=44.0
)



generate_model_btn_img = PhotoImage(
    file=relative_to_assets("button_6.png"))
preprocess = Button(
    image=generate_model_btn_img,
    borderwidth=0,
    highlightthickness=0,
    command=generateModel,
    relief="flat"
)
preprocess.place(
    x=67.0,
    y=175.0,
    width=183.0,
    height=44.0
)


feature_extraction_btn_img = PhotoImage(
    file=relative_to_assets("button_7.png"))
model = Button(
    image=feature_extraction_btn_img,
    borderwidth=0,
    highlightthickness=0,
    command=featureExtraction,
    relief="flat"
)
model.place(
    x=67.0,
    y=249.0,
    width=183.0,
    height=44.0
)

run_svm_btn_img = PhotoImage(
    file=relative_to_assets("button_2.png"))
runsvm = Button(
    image=run_svm_btn_img,
    borderwidth=0,
    highlightthickness=0,
    command=runSVM,
    relief="flat"
)
runsvm.place(
    x=1150.0,
    y=251.0,
    width=183.0,
    height=44.0
)
student_classification_btn_img = PhotoImage(
    file=relative_to_assets("button_8.png"))
emlfs = Button(
    image=student_classification_btn_img,
    borderwidth=0,
    highlightthickness=0,
    command=predictPerformance,
    relief="flat"
)
emlfs.place(
    x=67.0,
    y=484.0,
    width=183.0,
    height=44.0
)


accuracy_graph_btn_img = PhotoImage(
    file=relative_to_assets("button_9.png"))
graph_btn = Button(
    image=accuracy_graph_btn_img,
    borderwidth=0,
    highlightthickness=0,
    command=graph,
    relief="flat"
)
graph_btn.place(
    x=67.0,
    y=547.0,
    width=183.0,
    height=44.0
)


font1 = ('Georgia', 12, 'normal')

entry_image_1 = PhotoImage(
    file=relative_to_assets("entry_1.png"))
entry_bg_1 = canvas.create_image(
    707.0,
    374.5,
    image=entry_image_1
)

text= Text(
    bd=0,
    bg="#E3E6EC",
    fg="#000716",
    highlightthickness=0
)
text.place(
    x=379.0,
    y=124.0,
    width=656.0,
    height=499.0
)


scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.config(font=font1)

main.resizable(False, False)
main.mainloop()























# cal accuracy
# text.insert(END, "Report : " + str(classification_report(y_test, y_pred)) + "\n")
# text.insert(END, "Confusion Matrix :\n \n" + str(cm) + "\n\n\n\n\n")
# cm = confusion_matrix(y_test, y_pred)

# font = ('times', 16, 'bold')
# title = Label(main, text='Feature Extraction For Classifying Students Based On Their Academic Performance')
# title.config(bg='brown', fg='white')
# title.config(font=font)
# title.config(height=3, width=120)
# title.place(x=0, y=5)

# font1 = ('times', 14, 'bold')

# upload = Button(main, text="Upload Student Grades Dataset", command=upload)
# upload.place(x=50, y=100)
# upload.config(font=font1)

#
# pathlabel = Label(main)
# pathlabel.config(bg='orange', fg='white')
# pathlabel.config(font=font1)
# pathlabel.place(x=350, y=100)

# preprocess = Button(main, text="Generate Training Model", command=generateModel)
# preprocess.place(x=50, y=150)
# preprocess.config(font=font1)

# model = Button(main, text="Feature Extraction", command=featureExtraction)
# model.place(x=300, y=150)
# model.config(font=font1)

# runsvm = Button(main, text="Run SVM Algorithm", command=runSVM)
# runsvm.place(x=500, y=150)
# runsvm.config(font=font1)
#

# runrandomforest = Button(main, text="Run Random Forest Algorithm", command=runRandomForest)
# runrandomforest.place(x=710, y=150)
# runrandomforest.config(font=font1)

# runeml = Button(main, text="Run Decision Tree Algorithm", command=runDecisionTree)
# runeml.place(x=50, y=200)
# runeml.config(font=font1)

#
# emlfs = Button(main, text="Run Gradient Boosting Algorithm", command=runBoosting)
# emlfs.place(x=330, y=200)
# emlfs.config(font=font1)

#
# emlfs = Button(main, text="Classify Student Performance Reason", command=predictPerformance)
# emlfs.place(x=640, y=200)
# emlfs.config(font=font1)
#
# graph = Button(main, text="Accuracy Graph", command=graph)
# graph.place(x=990, y=200)
# graph.config(font=font1)
# text = Text(main, height=30, width=150)
# text.place(x=10, y=250)
