from pyhealth.metrics import binary_metrics_fn, multiclass_metrics_fn
import pandas as pd

pre_path = 'Weights'
model_name = 'abmil'
file_path = pre_path + '/' + model_name + '/UM.csv'
df = pd.read_csv(file_path)
y_true = df['correct_predictions']
y_pred = df['entropy']
auc = binary_metrics_fn(y_true, y_pred, metrics=['roc_auc'])
print(1-auc['roc_auc'])