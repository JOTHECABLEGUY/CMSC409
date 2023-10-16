# polt.py
	@ Dependencies: matplotlib.pyplot
		python -m pip install -U pip
		python -m pip install -U matplotlib
	@ Params: None
	@ return: None
	@ Usage: 
		This file must (and will) be in the same folder as the text files containing the car data sets (i.e groupA.txt, groupB.txt, and/or groupC.txt)
		Run this program with `python plot.py`
		The program will print the names of the available text file names, prefaced by a number. These numbers are keys for a dictionary within the program, so enter the number (a single number) that corresponds to the data file that you want to open. 
		The program will then open the indicated file, and proceed to populate a scatter plot. The program will output the True Positive, False Positive, False Negative, True Negative, Error, and Accuracy Rates to the Terminal. The program will then show a scatter plot with the (normalized) points provided by the data file with a line of separation denoting the classification line used in positive/negative (big/small) classifications.
		The program will exit once the plot window is closed by the user. The program can then be run again for any of the text files. 