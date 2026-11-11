# Supplemental wrist observation review

The smooth episode's exported frame 1184 maps to raw right-wrist source frame 1210; residual/hold-grip exported frame 1260 maps to raw frame 1260. Frame numbering must be stated explicitly. Detector reports use exported video zero-based indices and align phase with HDF5 monotonic timestamps and command CSV.

Local visual inspection of grip-hold frames 1100, 1255, 1260 and 1270 confirms the segmented yellow region is the plug housing. Its visible top moves down relative to the image/fingers during insertion. This supports appearance change, not a unique causal assignment to slip, collision or low friction. The image montage remains in ignored .local/ and is not published. The earliest detected threshold crossing is not guaranteed to be the earliest physical change.

With saturation minima 65/85/105, first candidate frames were smooth 1185/1184/1184, residual 1262/1260/1258, grip-hold 1255/1260/1260. All map to insert. These are threshold-sensitivity results, not independent labelled detection accuracy. No force/torque data was used or created.
