import numpy as np
import pandas as pd


def build_batch(df, batch_size):
  batch = df.sample(n=batch_size).copy() #copy 50 random
  batch.set_index(np.arange(batch_size), inplace=True)
  return batch

def predict_fare(model, df, features, label, batch_size=50):
  batch = build_batch(df, batch_size)
  predicted_values = model.predict_on_batch(x=batch.loc[:, features].values)

  data = {"Predicted_Performance": [], "Observed_Performance": [], "L1_LOSS": []}
  for i in range(batch_size):
    predicted = predicted_values[i][0]
    observed = batch.at[i, label]
    data["Predicted_Performance"].append(predicted)
    data["Observed_Performance"].append(observed)
    data["L1_LOSS"].append((abs(observed - predicted)))

  output_df = pd.DataFrame(data)
  return output_df

def show_predictions(output):
  header = "-" * 80
  banner = header + "\n" + "|" + "PREDICTIONS".center(78) + "|" + "\n" + header
  print(banner)
  print(output)
  return