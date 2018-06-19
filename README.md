# Gender_Classification
The software that is described in "Gender Prediction From Tweets With Convolutional Neural Networks" for PAN 2018

Installation
----------------------------------------------------

1) You can reach the code with `git clone https://github.com/Darg-Iztech/Gender_Classification.git`
2) In command line execute `pip install -r requirements.txt` to get dependent packages
3) You can download the dataset from https://pan.webis.de/clef18/pan18-web/author-profiling.html
4) Our software uses embeddings to get word ids and create vocabulary, that's why, an embedding trained in twitter (vocabulary of twitter is different than other formal ones) would be okay to use
5) You can run the code as `python main.py -i absolute_path_to_dataset -o absolute_path_of_output`
6) Software is portable with Python 2.7 and Python 3.x
7) If you have any problem with the code, feel free to share us.
