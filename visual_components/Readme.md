# License
<p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><a property="dct:title" rel="cc:attributionURL" href="https://github.com/DFKI-Interactive-Machine-Learning/ophthalmo-cdss">Ophthalmo-CDSS</a> by <a rel="cc:attributionURL dct:creator" property="cc:attributionName" href="https://github.com/robertleist">Robert Andreas Leist</a> is licensed under <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY-NC-SA 4.0<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/sa.svg?ref=chooser-v1" alt=""></a></p>

# Visual Components Documentation
This folder contains the documentation for the visual components of the project. The VCs are arranged as shown in the following figure:
<img src="../icons/VCs.png" alt="Visual Components of the Dashboard" width="700"/>
The following VCs are available:
- VC0 Sidebar:
  - Main functionality is patient selection. Selection widget save values to a key in the session state.
  - In this file, all dataframes are loaded and saved in the session state.
- VC1 Top bar:
  - Show Icons
  - Meta Data
  - Treatment Status
  - IVOM Timeline
  - NEW: Shows table of given IVOMs
- VC2 OCT Viewer:
  - VC2.1 Tools:
    - Selection of main and compare OCT
    - Selection of view: IRSLO, Slice or 3D
    - Toggle on Compare Slider, Alignment and Segmentations
    - Additional controls for IRSLO and Slice
  - Main view: Shows imaging data
- VC3 History Graphs:
  - Four tabs: Visual acuity, Volume of fluids, Volume of PED and IOP
  - Features Interactive line plots
- VC4 Metrics:
  - Show development from last to current visit
- VC5 Recommendation:
  - Show recommendation based on the current visit and the recommendation algorithm
  - Shows reasons for recommendation
- VC6 Info box:
  - Reasoning: Shows the reasoning behind the recommendation
  - Visit Diff: Shows the difference between the current and last visit
  - Mean Thickness: Shows the mean thickness of the current visit