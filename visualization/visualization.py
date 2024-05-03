from matplotlib import pyplot as plt

def visualize_speckles(x_train, save_path = 'speckles_sample.png', please_also_show = False):
  # Construct a figure on which we will visualize the images.
  fig, axes = plt.subplots(4, 5, figsize=(9, 8))

  # Plot each of the sequential images for one random data example.
  data_choice = np.random.choice(range(len(x_train)), size=1)[0]
  for idx, ax in enumerate(axes.flat):
      ax.imshow(np.squeeze(x_train[data_choice][idx]), cmap="gray")
      ax.set_title(f"Frame {idx + 1}")
      ax.axis("off")

  # Print information and display the figure.
  print(f"Displaying sample frames {data_choice}.")
  plt.savefig(save_path, bbox_inches='tight')
  plt.show()
  
