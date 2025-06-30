# LegoID
## Recognition and identification of LEGO sets from pictures

#### What is this project?
This project follows the development for an automatic LEGO set recognition tool based on image analysis and use of existing databases. The process takes as an input images of LEGO pieces and extracts features from all parts pictured like shape, color, size and stud distribution. This info is taken and compared against  an official database of sets and parts, previously curated for this specific project. The development implements a complete pipeline from image input to part recognition taking into account variability in piece orientation, looking for a working tool on as many environments as possible. These recognized pieces are forwarded to an identification system where possible errors in previous steps can be eased down, adding robustness. The results obtained show moderate accuracy and demonstrate the potential of the proposed tool.

#### Directory structure
* _data_: Required files for script execution.
  * _colors_hex.csv_
  * _complete_ref.csv_
  * _piece_combinations.csv_
* _img_: Directory with sample pictures of pieces.
* _classes_: Classes used on main script.
  * LegoPiece class 
* _funcions_: Files containing brick recognition functions separated by goal of the section
* _funcions_set_: Files containing set identification functions

#### How to use
* Make sure all required files from the _data_ directory are present.
* Execute main script _main.py_
* Choose an image for standard parts
![Standard window picture](https://i.imgur.com/r99NQEh.png)
* Choose an image for non-standard parts
* ![Non-standard window picture](https://i.imgur.com/LuznIaJ.png)
* Wait for the script to end, a progress percentage is shown.
* Check the results!
