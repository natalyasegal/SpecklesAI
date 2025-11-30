import numpy as np
import matplotlib.pyplot as plt

def create_image_with_2_comfusion_matrices(cm_1, cm_percent_1, cm_2, cm_percent_2):
  plt.rcParams.update({'font.size': 14})

  # --- Create subplots ---
  fig, axes = plt.subplots(ncols=2, figsize=(10, 4))

  # We will map both percentage matrices onto the SAME [0..1] scale
  # so color intensities are comparable across the two plots.

  # ========= Plot 1 =========
  im1 = axes[0].imshow(cm_percent_1, interpolation='nearest', cmap='Blues',
                      vmin=0, vmax=1)
  axes[0].set_title("Confusion Matrix (test, per chunk)", fontsize=12)
  axes[0].set_xticks([0, 1])
  axes[0].set_yticks([0, 1])
  axes[0].set_xticklabels(["0", "1"])
  axes[0].set_yticklabels(["0", "1"])

  # Annotate each cell with absolute count and percentage
  for i in range(2):
      for j in range(2):
          val_abs = cm_1[i, j]
          val_pct = cm_percent_1[i, j] * 100  # fraction → percent
          # If percent > 50%, use white text for contrast
          color = "white" if cm_percent_1[i, j] > 0.5 else "black"
          axes[0].text(
              j, i,
              f"{val_abs}\n({val_pct:.1f}%)",
              ha='center', va='center', color=color
          )

  axes[0].set_xlabel("Predicted label", fontsize=12)
  axes[0].set_ylabel("True label", fontsize=12)

  # ========= Plot 2 =========
  im2 = axes[1].imshow(cm_percent_2, interpolation='nearest', cmap='Blues',
                      vmin=0, vmax=1)
  axes[1].set_title("Confusion Matrix (test, 1 sec aggregation)", fontsize=12)
  axes[1].set_xticks([0, 1])
  axes[1].set_yticks([0, 1])
  axes[1].set_xticklabels(["0", "1"])
  axes[1].set_yticklabels(["0", "1"])

  for i in range(2):
      for j in range(2):
          val_abs = cm_2[i, j]
          val_pct = cm_percent_2[i, j] * 100
          color = "white" if cm_percent_2[i, j] > 0.5 else "black"
          axes[1].text(
              j, i,
              f"{val_abs}\n({val_pct:.1f}%)",
              ha='center', va='center', color=color
          )

  axes[1].set_xlabel("Predicted label", fontsize=12)
  axes[1].set_ylabel("True label", fontsize=12)

  plt.tight_layout()
  plt.show()
