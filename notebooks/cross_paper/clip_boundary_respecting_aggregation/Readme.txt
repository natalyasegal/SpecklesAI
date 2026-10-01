Results are reported for Wernicke's area using normalized inputs, 1 s windows, and 30 s calibration, respecting clip boundaries. 
Stride = 1 denotes sliding-window aggregation, producing overlapping 1 s windows, whereas stride = K denotes independent, non-overlapping 1 s inputs. 
FPS denotes the acquisition frame rate. 1s inputs correspond to aggregation of 25 consecutive 40-ms chunks, K = 25. Results obtained with stride = 1 were nearly 
identical whether clip boundaries were respected within the training, validation, and test sets (Supplementary Table 4) or not (Supplementary Table 1); in both cases, 
there was no leakage between these sets. In contrast, stride = K = 25 produced lower performance (AUC = 0.942 vs. 0.999; F1 = 0.923 vs. 0.994) 
and greater variability.

We also performed clustering while respecting clip boundaries. For Native vs. Swedish with 1 s inputs across 14 subjects, mean ARI = 0.959 and mean NMI = 0.948; substituting a different encoder for Subject 10 increased these to mean ARI = 0.988 and mean NMI = 0.982. These values are slightly higher than, those obtained without respecting clip boundaries, confirming that the reported clustering agreement is not an artifact of windows crossing clip boundaries.
