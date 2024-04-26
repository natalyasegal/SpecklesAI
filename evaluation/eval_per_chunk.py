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

def visualize_data(x_train):
  # Construct a figure on which we will visualize the images.
  fig, axes = plt.subplots(4, 5, figsize=(9, 8))

  # Plot each of the sequential images for one random data example.
  data_choice = np.random.choice(range(len(x_train)), size=1)[0]
  for idx, ax in enumerate(axes.flat):
      ax.imshow(np.squeeze(x_train[data_choice][idx]), cmap="gray")
      ax.set_title(f"Frame {idx + 1}")
      ax.axis("off")

  # Print information and display the figure.
  print(f"Displaying frames for example {data_choice}.")
  plt.show()
