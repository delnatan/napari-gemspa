# napari-gemspa

This plugin provides for analysis tools for data from single particle tracking experiments.  It provides an interface for particle localization and tracking using [trackpy](http://soft-matter.github.io/trackpy/dev/index.html).  It also allows for import of tracking data from Mosaic and Trackmate.  These files must be tab/comma delimited text files.  It provides an option to exclude particles/tracks masked with a labels layer.

There are 5 tabs available in the plugin, following the workflow of data analysis:

1) New/Open: open nd2/tiff time-lapse movie files and/or import a tracks layer (from Mosaic, Trackmate or napari-gemspa saved tracks layer)
2) Locate: locate particles with trackpy
3) Link: link particles with trackpy
4) Filter Links: filter links with trackpy
5) Analyze: Perform analysis on tracks from a tracks layer (can be from imported file from step 1 or layer created in step 3)

Detailed description of features:

1) New/Open
![1_1.png](screen_shots%2F1_1.png)

"Add layer" button will create a blank 2D (no time dimension) layer that is the same height/width as the currently selected image layer.  Alternatively, a labeled mask can be opened from a file.

Track files from other software or previously saved by GEMspa can also be imported in this pane.  Only tab/comma (.csv/.txt/.tsv) delimited text files are allowed.

GEMspa expects these columns in the header: ['track_id', 'frame', 'z', 'y', 'x']

Mosaic expects these columns in the header: ['Trajectory', 'Frame', 'z', 'y', 'x']

Trackmate expects these columns in the header: ['TRACK_ID', 'FRAME', 'POSITION_Z', 'POSITION_Y', 'POSITION_X'],
* 3 rows will be skipped for Trackmate files (assumes data begins at the 4th row after the header)

Trackpy expects these columns in the header: ['particle', 'frame', 'z', 'y', 'x']

(Case and order insensitive)

2) Locate
![2_1.png](screen_shots%2F2_1.png)

In this tab, adjust the parameters and perform particle localization with [trackpy.locate](http://soft-matter.github.io/trackpy/dev/generated/trackpy.locate.html#trackpy.locate).  To first test out parameters on a single frame, check the "Process only current frame" checkbox.  Please refer to the trackpy documentation for more details on parameters.

After localization is performed, a new points layer will be created and particles will be shown circled in red.  In the example, we have used a labels layer to exclude particles outside an ROI (this is optional):
![2_2.png](screen_shots%2F2_2.png)

In addition, the mass histogram and subpixel bias histograms will be shown for help with adjusting the mass and diameter parameters:
![2_3.png](screen_shots%2F2_3.png)

3) Link
![3_1.png](screen_shots%2F3_1.png)
In this tab, adjust parameters and perform linking with [trackpy.link](http://soft-matter.github.io/trackpy/dev/generated/trackpy.link.html).  Once linking is performed a new tracks layer will be added.  Please refer to the trackpy documentation for more details on parameters.

In addition, scatter plots of mass vs. size and mass vs. eccentricity, as well as the track lengths histogram are shown for help with filtering tracks. (next step)
![3_2.png](screen_shots%2F3_2.png)

4) Filter
![4_1.png](screen_shots%2F4_1.png)
In this tab, adjust parameters and filter links from trackpy output.

5) Analyze
![5_1.png](screen_shots%2F5_1.png)
In this tab, adjust parameters and perform analysis.
