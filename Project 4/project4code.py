#!/usr/bin/env python

"""Porter Stemming Algorithm
This is the Porter stemming algorithm, ported to Python from the
version coded up in ANSI C by the author. It may be be regarded
as canonical, in that it follows the algorithm presented in

Porter, 1980, An algorithm for suffix stripping, Program, Vol. 14,
no. 3, pp 130-137,

only differing from it at the points maked --DEPARTURE-- below.

See also http://www.tartarus.org/~martin/PorterStemmer

The algorithm as described in the paper could be exactly replicated
by adjusting the points of DEPARTURE, but this is barely necessary,
because (a) the points of DEPARTURE are definitely improvements, and
(b) no encoding of the Porter stemmer I have seen is anything like
as exact as this version, even with the points of DEPARTURE!

Vivake Gupta (v@nano.com)

Release 1: January 2001

Further adjustments by Santiago Bruno (bananabruno@gmail.com)
to allow word input not restricted to one word per line, leading
to:

release 2: July 2008
"""

import sys
import re
import pandas as pd
import math as m

class PorterStemmer:

    def __init__(self):
        """The main part of the stemming algorithm starts here.
        b is a buffer holding a word to be stemmed. The letters are in b[k0],
        b[k0+1] ... ending at b[k]. In fact k0 = 0 in this demo program. k is
        readjusted downwards as the stemming progresses. Zero termination is
        not in fact used in the algorithm.

        Note that only lower case sequences are stemmed. Forcing to lower case
        should be done before stem(...) is called.
        """

        self.b = ""  # buffer for word to be stemmed
        self.k = 0
        self.k0 = 0
        self.j = 0   # j is a general offset into the string

    def cons(self, i):
        """cons(i) is TRUE <=> b[i] is a consonant."""
        if self.b[i] == 'a' or self.b[i] == 'e' or self.b[i] == 'i' or self.b[i] == 'o' or self.b[i] == 'u':
            return 0
        if self.b[i] == 'y':
            if i == self.k0:
                return 1
            else:
                return (not self.cons(i - 1))
        return 1

    def m(self):
        """m() measures the number of consonant sequences between k0 and j.
        if c is a consonant sequence and v a vowel sequence, and <..>
        indicates arbitrary presence,

           <c><v>       gives 0
           <c>vc<v>     gives 1
           <c>vcvc<v>   gives 2
           <c>vcvcvc<v> gives 3
           ....
        """
        n = 0
        i = self.k0
        while 1:
            if i > self.j:
                return n
            if not self.cons(i):
                break
            i = i + 1
        i = i + 1
        while 1:
            while 1:
                if i > self.j:
                    return n
                if self.cons(i):
                    break
                i = i + 1
            i = i + 1
            n = n + 1
            while 1:
                if i > self.j:
                    return n
                if not self.cons(i):
                    break
                i = i + 1
            i = i + 1

    def vowelinstem(self):
        """vowelinstem() is TRUE <=> k0,...j contains a vowel"""
        for i in range(self.k0, self.j + 1):
            if not self.cons(i):
                return 1
        return 0

    def doublec(self, j):
        """doublec(j) is TRUE <=> j,(j-1) contain a double consonant."""
        if j < (self.k0 + 1):
            return 0
        if (self.b[j] != self.b[j-1]):
            return 0
        return self.cons(j)

    def cvc(self, i):
        """cvc(i) is TRUE <=> i-2,i-1,i has the form consonant - vowel - consonant
        and also if the second c is not w,x or y. this is used when trying to
        restore an e at the end of a short  e.g.

           cav(e), lov(e), hop(e), crim(e), but
           snow, box, tray.
        """
        if i < (self.k0 + 2) or not self.cons(i) or self.cons(i-1) or not self.cons(i-2):
            return 0
        ch = self.b[i]
        if ch == 'w' or ch == 'x' or ch == 'y':
            return 0
        return 1

    def ends(self, s):
        """ends(s) is TRUE <=> k0,...k ends with the string s."""
        length = len(s)
        if s[length - 1] != self.b[self.k]:  # tiny speed-up
            return 0
        if length > (self.k - self.k0 + 1):
            return 0
        if self.b[self.k-length+1:self.k+1] != s:
            return 0
        self.j = self.k - length
        return 1

    def setto(self, s):
        """setto(s) sets (j+1),...k to the characters in the string s, readjusting k."""
        length = len(s)
        self.b = self.b[:self.j+1] + s + self.b[self.j+length+1:]
        self.k = self.j + length

    def r(self, s):
        """r(s) is used further down."""
        if self.m() > 0:
            self.setto(s)

    def step1ab(self):
        """step1ab() gets rid of plurals and -ed or -ing. e.g.

           caresses  ->  caress
           ponies    ->  poni
           ties      ->  ti
           caress    ->  caress
           cats      ->  cat

           feed      ->  feed
           agreed    ->  agree
           disabled  ->  disable

           matting   ->  mat
           mating    ->  mate
           meeting   ->  meet
           milling   ->  mill
           messing   ->  mess

           meetings  ->  meet
        """
        if self.b[self.k] == 's':
            if self.ends("sses"):
                self.k = self.k - 2
            elif self.ends("ies"):
                self.setto("i")
            elif self.b[self.k - 1] != 's':
                self.k = self.k - 1
        if self.ends("eed"):
            if self.m() > 0:
                self.k = self.k - 1
        elif (self.ends("ed") or self.ends("ing")) and self.vowelinstem():
            self.k = self.j
            if self.ends("at"):
                self.setto("ate")
            elif self.ends("bl"):
                self.setto("ble")
            elif self.ends("iz"):
                self.setto("ize")
            elif self.doublec(self.k):
                self.k = self.k - 1
                ch = self.b[self.k]
                if ch == 'l' or ch == 's' or ch == 'z':
                    self.k = self.k + 1
            elif (self.m() == 1 and self.cvc(self.k)):
                self.setto("e")

    def step1c(self):
        """step1c() turns terminal y to i when there is another vowel in the stem."""
        if (self.ends("y") and self.vowelinstem()):
            self.b = self.b[:self.k] + 'i' + self.b[self.k+1:]

    def step2(self):
        """step2() maps double suffices to single ones.
        so -ization ( = -ize plus -ation) maps to -ize etc. note that the
        string before the suffix must give m() > 0.
        """
        if self.b[self.k - 1] == 'a':
            if self.ends("ational"):
                self.r("ate")
            elif self.ends("tional"):
                self.r("tion")
        elif self.b[self.k - 1] == 'c':
            if self.ends("enci"):
                self.r("ence")
            elif self.ends("anci"):
                self.r("ance")
        elif self.b[self.k - 1] == 'e':
            if self.ends("izer"):
                self.r("ize")
        elif self.b[self.k - 1] == 'l':
            if self.ends("bli"):
                self.r("ble")  # --DEPARTURE--
            # To match the published algorithm, replace this phrase with
            #   if self.ends("abli"):      self.r("able")
            elif self.ends("alli"):
                self.r("al")
            elif self.ends("entli"):
                self.r("ent")
            elif self.ends("eli"):
                self.r("e")
            elif self.ends("ousli"):
                self.r("ous")
        elif self.b[self.k - 1] == 'o':
            if self.ends("ization"):
                self.r("ize")
            elif self.ends("ation"):
                self.r("ate")
            elif self.ends("ator"):
                self.r("ate")
        elif self.b[self.k - 1] == 's':
            if self.ends("alism"):
                self.r("al")
            elif self.ends("iveness"):
                self.r("ive")
            elif self.ends("fulness"):
                self.r("ful")
            elif self.ends("ousness"):
                self.r("ous")
        elif self.b[self.k - 1] == 't':
            if self.ends("aliti"):
                self.r("al")
            elif self.ends("iviti"):
                self.r("ive")
            elif self.ends("biliti"):
                self.r("ble")
        elif self.b[self.k - 1] == 'g':  # --DEPARTURE--
            if self.ends("logi"):
                self.r("log")
        # To match the published algorithm, delete this phrase

    def step3(self):
        """step3() dels with -ic-, -full, -ness etc. similar strategy to step2."""
        if self.b[self.k] == 'e':
            if self.ends("icate"):
                self.r("ic")
            elif self.ends("ative"):
                self.r("")
            elif self.ends("alize"):
                self.r("al")
        elif self.b[self.k] == 'i':
            if self.ends("iciti"):
                self.r("ic")
        elif self.b[self.k] == 'l':
            if self.ends("ical"):
                self.r("ic")
            elif self.ends("ful"):
                self.r("")
        elif self.b[self.k] == 's':
            if self.ends("ness"):
                self.r("")

    def step4(self):
        """step4() takes off -ant, -ence etc., in context <c>vcvc<v>."""
        if self.b[self.k - 1] == 'a':
            if self.ends("al"):
                pass
            else:
                return
        elif self.b[self.k - 1] == 'c':
            if self.ends("ance"):
                pass
            elif self.ends("ence"):
                pass
            else:
                return
        elif self.b[self.k - 1] == 'e':
            if self.ends("er"):
                pass
            else:
                return
        elif self.b[self.k - 1] == 'i':
            if self.ends("ic"):
                pass
            else:
                return
        elif self.b[self.k - 1] == 'l':
            if self.ends("able"):
                pass
            elif self.ends("ible"):
                pass
            else:
                return
        elif self.b[self.k - 1] == 'n':
            if self.ends("ant"):
                pass
            elif self.ends("ement"):
                pass
            elif self.ends("ment"):
                pass
            elif self.ends("ent"):
                pass
            else:
                return
        elif self.b[self.k - 1] == 'o':
            if self.ends("ion") and (self.b[self.j] == 's' or self.b[self.j] == 't'):
                pass
            elif self.ends("ou"):
                pass
            # takes care of -ous
            else:
                return
        elif self.b[self.k - 1] == 's':
            if self.ends("ism"):
                pass
            else:
                return
        elif self.b[self.k - 1] == 't':
            if self.ends("ate"):
                pass
            elif self.ends("iti"):
                pass
            else:
                return
        elif self.b[self.k - 1] == 'u':
            if self.ends("ous"):
                pass
            else:
                return
        elif self.b[self.k - 1] == 'v':
            if self.ends("ive"):
                pass
            else:
                return
        elif self.b[self.k - 1] == 'z':
            if self.ends("ize"):
                pass
            else:
                return
        else:
            return
        if self.m() > 1:
            self.k = self.j

    def step5(self):
        """step5() removes a final -e if m() > 1, and changes -ll to -l if
        m() > 1.
        """
        self.j = self.k
        if self.b[self.k] == 'e':
            a = self.m()
            if a > 1 or (a == 1 and not self.cvc(self.k-1)):
                self.k = self.k - 1
        if self.b[self.k] == 'l' and self.doublec(self.k) and self.m() > 1:
            self.k = self.k - 1

    def stem(self, p, i, j):
        """In stem(p,i,j), p is a char pointer, and the string to be stemmed
        is from p[i] to p[j] inclusive. Typically i is zero and j is the
        offset to the last character of a string, (p[j+1] == '\0'). The
        stemmer adjusts the characters p[i] ... p[j] and returns the new
        end-point of the string, k. Stemming never increases word length, so
        i <= k <= j. To turn the stemmer into a module, declare 'stem' as
        extern, and delete the remainder of this file.
        """
        # copy the parameters into statics
        self.b = p
        self.k = j
        self.k0 = i
        if self.k <= self.k0 + 1:
            return self.b  # --DEPARTURE--

        # With this line, strings of length 1 or 2 don't go through the
        # stemming process, although no mention is made of this in the
        # published algorithm. Remove the line to match the published
        # algorithm.

        self.step1ab()
        self.step1c()
        self.step2()
        self.step3()
        self.step4()
        self.step5()
        return self.b[self.k0:self.k+1]

def dist(pattern, cluster):

    # return Euclidean distance between 2 points of arbitrary dimensionality
    return m.sqrt(sum([pow(pattern[num] - cluster[num], 2) for num in range(len(pattern))]))

if __name__ == '__main__':

    # call stemmer contructor
    p = PorterStemmer()

    # get the stop words
    stop_words = open("Project4_stop_words.txt").read().splitlines()

    # if any arguments are provided (the project4code.py is argv[0]), then program can proceed
    if len(sys.argv) > 3:

        # for each file, since we can repeat for multiple input files
        for f in sys.argv[3:]:

            # arbitrary frequency threshold
            frequency_threshold = int(sys.argv[1])

            # open the input file and get contents
            with open(f, 'r') as infile:
                contents = infile.read().lower()

            # split the file into separate "paragraphs" when each is separated by a newline char
            paragraphs = contents.split('\n')

            # filter out empty lines
            filtered_paragraphs = [para for para in paragraphs if para]

            # this will hold the output from stemming and tokenization (list of lists containing stemmed and tokenized input)
            stemmed_tokenized_paragraphs = []

            # for each paragraph in the input file
            for paragraph in filtered_paragraphs:

                # take out html tags, as well as special chars using regex
                paragraph = re.sub(r'<br />', r'', paragraph )
                paragraph = re.sub(r'[^A-Za-z ]+', '', paragraph)

                # split the paragraph into unstemmed words
                paragraph = paragraph.split(' ')

                # this removes stop words, thus every word in tokens is a valid token
                tokens = [word for word in paragraph if word not in stop_words]

                # stem the tokens to get list of root words
                stemmed_tokens = [p.stem(t, 0, len(t)-1) for t in tokens if t]

                # add the list of stemmed words for current paragraph to the list of tokenized paragraphs
                stemmed_tokenized_paragraphs.append(stemmed_tokens)

            # hold all words, will be whittled down
            all_words = []

            # frequency of all words by paragraph
            paragraph_freq_dict = {}

            # frequency of all words across the entire document
            whole_document_freq_dict = {}
            
            # loop through all paragraphs by index
            for stp in range(len(stemmed_tokenized_paragraphs)):

                # make a dictionary for the current paragraph
                paragraph_freq_dict[stp] = {}

                # for each word in the current paragraph
                for word in stemmed_tokenized_paragraphs[stp]:

                    # add the word to the list of all words
                    all_words.append(word)

                    # if first occurrence in the whole file, initialize the frequency to 0
                    if word not in whole_document_freq_dict.keys():
                        whole_document_freq_dict[word] = 0

                    # if first occurrence in the paragraph, initialize the frequency to 0
                    if word not in paragraph_freq_dict[stp].keys():
                        paragraph_freq_dict[stp][word] = 0

                    # increment frequencies in the whole document and the paragraphs scopes
                    paragraph_freq_dict[stp][word] += 1
                    whole_document_freq_dict[word] += 1

            # gets all unique words using sets (sets do not contain duplicates)
            whole_vect = list(set(all_words))

            # filter the unique words by the frequency threshold set above (this frequency threshold applies to whole document frequency, 
            #   not paragraph specific). This will serve as the header in the output table
            final_feature_vector = [word for word in whole_vect if whole_document_freq_dict[word] > frequency_threshold]

            # initialize the table to be a list of lists (2D array)
            table_array = [[] for i in range(len(stemmed_tokenized_paragraphs))]
            
            # for each paragraph indexed by number
            for stp in range(len(stemmed_tokenized_paragraphs)):

                # populate the table by checking for each word in each paragraph's frequency dictionary.
                #   Table is formatted [para_number][word_frequency]
                for word in final_feature_vector:
                    try:

                        # check the paragraph frequency dictionary for the frequency of the word
                        table_array[stp].append(paragraph_freq_dict[stp][word])

                    except Exception as e:

                        # if word doesnt exist in the paragraph, exception is thrown: Key Error. Handled here to reflect 0 occurrences of the word
                        table_array[stp].append(0)

            # convert table to a DataFrame (looks nicer than CSV file) with column names (feature vector) and the proper row naming (by number)
            df = pd.DataFrame(table_array, columns = final_feature_vector, index = [f'Paragraph {i}' for i in range(1, len(stemmed_tokenized_paragraphs)+1)])
            
            # export data to CSV format for user viewing
            df.to_csv('Processed_text.csv')

            # define learning rate and distance threshold
            a = 1
            rad_max = int(sys.argv[2])

            # the below lines/comments treat paragraphs and pattern interchangably. Also, COG stands for Center Of Gravity, AKA centroid
            
            # build patterns list to be input to clustering (each row of TDM)
            patterns = [row for row in table_array]

            # dictionary to store mapping of each paragraph to the cluster it belongs to
            cluster_map = {}

            # list to store the centers of gravity for the clusters that are formed
            clusters = []


            for pattern in range(len(patterns)):

                # if no clusters have been formed
                if len(clusters) == 0:

                    # make a new cluster and add it to the list of clusters. The [1] adds the element 1 to the end of the list. This element serves
                    # as a counter to keep track of how many elements are already in the cluster (how many patterns have been used to adjust the cluster's
                    # center of gravity)
                    clusters.append(patterns[pattern] + [1])

                    # set the index of this pattern's cluster to 0
                    i = 0

                else:

                    # find distances from this pattern to all currently formed clusters
                    distances = [dist(patterns[pattern], clusters[cluster][:-1]) for cluster in range(len(clusters))]

                    # get the lowest distance from the pattern to a cluster
                    min_dist = min(distances)

                    # get the index of the closest cluster (this is the location of the closest cluster within the clusters array)
                    min_cluster = distances.index(min_dist)

                    # check if the minimum distance is below the distance threshold
                    if min_dist > rad_max:

                        # if it isn't, make a new cluster
                        clusters.append(patterns[pattern] + [1])

                        # set the index of this pattern's designated cluster to the last index of the clusters array
                        i = len(clusters) - 1

                    else:

                        # get the number of patterns that have modified the closest cluster's center of gravity
                        x = clusters[min_cluster][-1]

                        # increment the above value by 1 and store it for future use
                        clusters[min_cluster][-1] += 1

                        # update each element in the closest cluster's center of gravity according to the function 
                        # w_cluster = m*w_cluster + alpha*pattern
                        #             ---------------------------
                        #                       m+1
                        # where w_cluster is the weights of the center of gravity of the cluster closest to the current pattern,
                        #   m is the number of patterns already mapped to the closest cluster (how many patterns were used to adjust the
                        #   cluster's COG up to this point), alpha is a predetermined learning constant, and pattern is the current input 
                        #   pattern being processed. 
                        # Note: range is decremented by 1 to avoid modifying the last value in the cluster, 
                        # which is reserved for keeping track of how many patterns have been used to adjust the cluster's COG
                        for index in range(len(clusters[min_cluster]) -1):
                            clusters[min_cluster][index] = (x * clusters[min_cluster][index] + a * patterns[pattern][index]) / (x + 1)

                        # set the index of this pattern's designated cluster to the index of the closest cluster
                        i = min_cluster

                # every cluster gets a list in the dictionary. If the cluster has not been seen before, a new list is made for it. 
                #   Note: i+1 is used because starting at cluster 1 looks better. L + ratio
                if f'Cluster {i+1}' not in cluster_map.keys():
                    cluster_map[f'Cluster {i+1}'] = []

                # since a list was made in the previous statement, we can safely append the pattern to the appropriate cluster map
                cluster_map[f'Cluster {i+1}'].append(pattern)

            print(cluster_map)
    else:
        print(f'\n Error with input:\n\tUSAGE:\tpython project4code.py [frequency_threshold] [max_radius] [filename_1.txt filename_2.txt ... filename_n.txt]\n\t\t where filenames are .txt files containing '
                + 'documents that you want to cluster')
