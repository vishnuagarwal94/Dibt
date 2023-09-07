import pandas
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
dib=pandas.read_csv(r"C:\Users\dell\Desktop\Stream\dataset.csv")
Dec=DecisionTreeClassifier()
print(dib.head(5))
print(dib.info())
print(dib.describe())
print(dib.corr().T)
print(dib.isnull().sum())
dib.rename(columns={'Pregnancies' : 'Prg','Glucose' : 'Glu','BloodPressure' : 'BP','SkinThickness' : 'ST','Insulin' : 'Ins','DiabetesPedigreeFunction' : 'DPF'})
dib.columns=['Prg','Glu','BP','ST','Ins','BMI','DPF','Age','Outcome']
data=['Prg','Glu','BP','ST','Ins','BMI','DPF','Age']
x=dib[data]
y=dib['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
print(x_train.head(5))
print(y_train.head(5))
Dec.fit(x_train,y_train)
Dec.predict(x_test)
print(Dec.score(x_test,y_test))