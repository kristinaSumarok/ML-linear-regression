import pandas as pd
import Predict_Perfomance as Pr
import ModelTraining as Mt

df = pd.read_csv("Student_Performance.csv")
df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})

pd.set_option('display.max_columns', None)  # Show all columns

 #correlation
print(df.corr())

# Adjust these hyperparameters to see how they impact a training run.
learning_rate = 0.001
epochs = 20
batch_size = 50

features = ['Extracurricular Activities', 'Hours Studied', 'Previous Scores', 'Sleep Hours', 'Sample Question Papers Practiced']
label = 'Performance Index'

model_1 = Mt.run_experiment(df, features, label, learning_rate, epochs, batch_size)
output = Pr.predict_fare(model_1, df, features, label)
Pr.show_predictions(output)
