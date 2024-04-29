from plot_metric.functions import BinaryClassification

def plot_nice_roc_curve(x_test, y_test, model):
  # Visualisation with plot_metric
  y_true = y_test.flatten()
  y_test_predicted = model.predict(x_test)
  y_pred = y_test_predicted.flatten()
  bc = BinaryClassification(y_test, y_pred, labels=["Class 1", "Class 2"])

  # Figures
  plt.figure(figsize=(5,5))
  bc.plot_roc_curve()
  plt.show()

def display_confusion_matrix(th):
  np.set_printoptions(precision=2)
  titles_options = [("Confusion matrix, without normalization", None), ("Normalized confusion matrix", "true"),]
  y_test_predicted = model.predict(x_test)
  for title, normalize in titles_options:
      disp = ConfusionMatrixDisplay.from_predictions(y_test.flatten(), y_test_predicted.flatten() >th,cmap=plt.cm.Blues,normalize=normalize)
      disp.ax_.set_title(title)
      print(title)
      print(disp.confusion_matrix)
  plt.show()

