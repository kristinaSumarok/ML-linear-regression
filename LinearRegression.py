import seaborn as sns
import pandas as pd
import ModelTraining as mt

# @title
chicago_taxi_dataset = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv")

#@title Code - Read dataset

# Updates dataframe to use specific columns.
training_df = chicago_taxi_dataset[['TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'TIP_RATE']]

print(training_df.corr(numeric_only = True))

#@title Code - View pairplot
pair = sns.pairplot(training_df, x_vars=["FARE", "TRIP_MILES", "TRIP_SECONDS"], y_vars=["FARE", "TRIP_MILES", "TRIP_SECONDS"])
pair.savefig("pairplot.png")

# TODO - Adjust these hyperparameters to see how they impact a training run.
learning_rate = 0.001
epochs = 20
batch_size = 50

# Specify the feature and the label.
training_df.loc[:,'TRIP_MINUTES'] = training_df['TRIP_SECONDS']/60
features = ['TRIP_MILES','TRIP_MINUTES']
label = 'FARE'

model_1 = mt.run_experiment(training_df, features, label, learning_rate, epochs, batch_size)