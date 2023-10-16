###
	project4code.py
###

##
	by Anna, Christopher, and Jordan
## 

#
	@ Dependencies: matplotlib.pyplot
        python -m pip install -U pip
        python -m pip install -U math
        python -m pip install -U pandas
    @ Params: None
    @ return: None
#

#
	command: python project4code.py [frequency_threshold] [max_radius] [file_name]+
		[frequency_threshold]	: integer representing the minimum number of occurrences that a word must reach in order to be included in 								the Term-Document Matrix

		[max_radius] 			: integer representing the maximum distance a word can be from a center of an existing cluster to be 									considered a member of that cluster

		[file_name]+ 			: 1 or more filenames (can be relative or absolute paths) that specify which files to be scanned and 									clustered
#
#
	example: python.exe project4code.py 50 10 Project4_paragraphs.txt

		here, frequency_threshold is 50, max_radius is 10, and the file being processed is Project4_paragraphs.txt

	the learning constant, alpha, is currently hard set to 1, but the user can change it by editing line 531 in project4code.py

	during the run of the program, a file named "Processed_text.csv" is created and stored in the same folder as the source code. This file contains the TDM, with the feature vector contained within the first row, as it designates the columns. 
#
