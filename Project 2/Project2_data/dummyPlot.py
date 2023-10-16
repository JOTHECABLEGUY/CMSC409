from matplotlib.pyplot import subplots, show
import random
import os
import math
from sys import maxsize

# Functions*****************************************

# Returns the output of a soft bipolar activation
def f_unipolar(net, k = math.pow(10, -2)):
    return 1 / (1 + math.exp(-k * net) )


# Prints out data learned
def printdata(iteration, pattern, net, error, learn, arrayOfWeights):
    arrayOfWeights_formatted = ['%.5f' % elem for elem in arrayOfWeights]
    print("ITERATION = ", iteration,
    "pattern = ", pattern,
    "net = ", round(net, 5),
    "error = ", error,
    "learn = ", learn,
    "weights = ", arrayOfWeights_formatted)

# function to normalize data
def normalize(x, minimum, maximum):
    numerator = x - minimum
    denominator = maximum - minimum
    return float(numerator/denominator)

# Used for hard activation
def sign(net):
    return 1 if net >= 0 else 0

# Training neuron using hard activation
def hard_train(group, patterns, trainingSize):
    if group == "A":
        alpha = 0.2
        te_thresh = 10e-5
    elif group == "B":
        alpha = 0.00118
        te_thresh = 40
    else:
        if trainingSize == 1:
            alpha = 0.00002
            te_thresh = 800
        elif trainingSize == 2:
            alpha = 0.01
            te_thresh = 330
        
    min_e = maxsize
    weights = [random.random()-0.5, random.random()-0.5,random.random()-0.5]

    desired_out = [p[3] for p in patterns]
    num_patterns = len(patterns)
    num_weights = 3
    iterations = 5000
    final_te = 0.0
    for iteration in range(iterations):  # number of training cycles
        # temporary array to store the output
        actual = [0]*num_patterns
        te = 0.0
        for pattern in range(num_patterns):  # for all patterns (training data)
            net = 0
            for i in range(num_weights):  # for all inputs
                net += weights[i]*patterns[pattern][i]
            actual[pattern] = sign(net)
            error = desired_out[pattern] - actual[pattern]
            te += math.pow(error, 2)
            learn = alpha*error
            for i in range(num_weights):
                weights[i] += learn * patterns[pattern][i]
        if te < te_thresh:
            final_te = te
            min_e = final_te
            return [final_te, weights, iteration, min_e]
        if te < min_e:
            min_e = te
        if iteration == 4999:
            final_te = te
    return [final_te, weights, iteration, min_e]

# Training Neuron using soft activation
def soft_train(group, patterns, trainingSize):
    if group == "A":
        alpha = 0.2
        k = 0.2
        te_thresh = 10e-5
    elif group == "B":
        alpha = 0.013
        k = 0.01
        te_thresh = 40
    else:
        if trainingSize == 1:
            alpha = 0.00002
            k = 0.2
            te_thresh = 700
        elif trainingSize == 2:
            alpha = 0.01
            k = 5
            te_thresh = 177
        
    min_e = maxsize
    weights = [random.random()-0.5, random.random()-0.5,random.random()-0.5]
    desired_out = [p[3] for p in patterns]
    num_patterns = len(patterns)
    num_weights = 3
    iterations = 5000
    final_te = 0.0
    for iteration in range(iterations):  # number of training cycles
        # temporary array to store the output
        actual = [0]*num_patterns
        te = 0.0
        for pattern in range(num_patterns):  # for all patterns (training data)
            net = 0
            for i in range(num_weights):  # for all inputs
                net += weights[i]*patterns[pattern][i]
            actual[pattern] = f_unipolar(net, k)
            error = desired_out[pattern] - actual[pattern]
            # print(error)
            te += math.pow(error, 2)
            learn = alpha*error
            for i in range(num_weights):
                weights[i] += learn * patterns[pattern][i]
        if te < te_thresh:
            final_te = te
            min_e = final_te
            return [final_te, weights, iteration, min_e]
        if te < min_e:
            min_e = te
        if iteration == 4999:
            final_te = te
    return [final_te, weights, iteration, min_e]

# End of Functions***********************************


# initialize lists
car_weights, costs, colors, normalized_costs, normalized_car_weights = [
], [], [], [], []
# find text files
files = [f'{os.getcwd()}/{f}' for f in os.listdir(os.getcwd())
         if f.endswith(".txt")]
fig, ax = subplots()  # get subplots
fileDict = dict()  # init dict to print files

# build filename dictionary
for i in range(0, len(files)):
    fileDict[i+1] = files[i]

# print the dictionary for user to see
for key, value in fileDict.items():
    val = value.split("/")[-1]
    print(f"{key}: {val}")

# store input from user to use as a key
fileChoice = int(input("Enter the NUMBER corresponding to the file you want to open: "))

# read file
with open(fileDict[fileChoice]) as f:
    contents = f.readlines()
    group_name = f.name.split('/')[-1][:-4]  # format name of group

activationType = int(input("Enter 1 to use hard activation function.\nEnter 2 to use soft activation function.\nChoice: "))
trainingChoice = int(input("Enter 1 to train 75%\nEnter 2 to train 25%\nChoice: "))
plotChoice = int(input("Enter 1 to plot trained points\nEnter 2 to plot tested points\nChoice: "))

# use contents of text file to build coordinate lists
for line in contents:
    line = line.strip().split(",")      # format each line from file
    # configure color based on if 'small' or 'big'
    color = 'r' if line[2] == '0' else 'b'
    weight = float(line[1])
    cost = float(line[0])
    car_weights.append(weight)
    costs.append(cost)
    colors.append(color)

# set maxima and minima
maxCost = max(costs)
minCost = min(costs)
maxWeight = max(car_weights)
minWeight = min(car_weights)

big_list = []  # Holds the big, blue cars
small_list = []  # Holds the small, red cars

# normalize both weight and cost, then put them into lists separating blue cars from red cars
for weight, cost, color in zip(car_weights, costs, colors):
    normalized_car_weight = normalize(weight, minWeight, maxWeight)
    normalized_car_weights.append(normalized_car_weight)
    normalized_cost = normalize(cost, minCost, maxCost)
    normalized_costs.append(normalized_cost)
    if color == 'b':
        big_list.append([normalized_car_weight, normalized_cost, color])
    else:
        small_list.append([normalized_car_weight, normalized_cost, color])
# At this point, we have a list with all the blue cars and a list with the red cars

# These will hold the big blue cars
high_big_x_list = list()
high_big_y_list = list()
high_big_c_list = list()
high_big_d_list = list()
low_big_x_list = list()
low_big_y_list = list()
low_big_c_list = list()
low_big_d_list = list()

# These will hold the small red cars
high_small_x_list = list()
high_small_y_list = list()
high_small_c_list = list()
high_small_d_list = list()
low_small_x_list = list()
low_small_y_list = list()
low_small_c_list = list()
low_small_d_list = list()

# Split the blue cars into a 75% and 25% split
for i in range(len(big_list)):
    if not (i % 4 == 0):  # 75%
        high_big_x_list.append(big_list[i][0])
        high_big_y_list.append(big_list[i][1])
        high_big_c_list.append(big_list[i][2])
        high_big_d_list.append(1)
    else:  # 25%
        low_big_x_list.append(big_list[i][0])
        low_big_y_list.append(big_list[i][1])
        low_big_c_list.append(big_list[i][2])
        low_big_d_list.append(1)

# Split the red cars into a 75% and 25% split
for i in range(len(small_list)):  
    if not (i % 4 == 0):  # 75%
        high_small_x_list.append(small_list[i][0])
        high_small_y_list.append(small_list[i][1])
        high_small_c_list.append(small_list[i][2])
        high_small_d_list.append(0)
    else:  # 25%
        low_small_x_list.append(small_list[i][0])
        low_small_y_list.append(small_list[i][1])
        low_small_c_list.append(small_list[i][2])
        low_small_d_list.append(0)

# Lists out all data into pattern form to permit training
if trainingChoice == 1:  # Training 75%
    patterns = [[x, y, 1, d] for x, y, d in zip(high_big_x_list, high_big_y_list, high_big_d_list)] + [
    [x, y, 1, d] for x, y, d in zip(high_small_x_list, high_small_y_list, high_small_d_list)]

elif trainingChoice == 2:  # Training 25%
    patterns = [[x, y, 1, d] for x, y, d in zip(low_big_x_list, low_big_y_list, low_big_d_list)] + [
        [x, y, 1, d] for x, y, d in zip(low_small_x_list, low_small_y_list, low_small_d_list)]

random.shuffle(patterns)

if activationType == 1:
    output = hard_train(group_name[-1].upper(), patterns, trainingChoice)
elif activationType == 2:
    output = soft_train(group_name[-1].upper(), patterns, trainingChoice)

# weights = output[1]
weights = [0.8856168830140048, 1.5389426600846847, -1.210491379605038]
print(f'total error = {output[0]}\nnew weights = {weights}\nfound on iteration {output[2]}\nminimum error found = {output[-1]}')

# classification line
y_0 = weights[2]/weights[1]
x_0 = weights[2]/weights[0]
ax.plot([abs(y_0), 0], [0, abs(x_0)])

reg_line = weights
# # init confusion vars
true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0

# Makes sure to use the correct data points that are being trained
if trainingChoice == 1:
    big_x_list = high_big_x_list
    big_y_list = high_big_y_list
    small_x_list = high_small_x_list
    small_y_list = high_small_y_list
elif trainingChoice == 2:
    big_x_list = low_big_x_list
    big_y_list = low_big_y_list
    small_x_list = low_small_x_list
    small_y_list = low_small_y_list


# run classification line
for car_weight, cost, color in zip(big_x_list, big_y_list, ['b']*len(big_y_list)):
    total = car_weight * reg_line[0] + cost * reg_line[1] + reg_line[2]

    if total >= 0:
        if color == 'b':
            true_pos += 1
        else:
            false_neg += 1
    else:
        if color == 'r':
            true_neg += 1
        else:
            false_pos += 1

for car_weight, cost, color in zip(small_x_list, small_y_list, ['r']*len(small_x_list)):
    total = car_weight * reg_line[0] + cost * reg_line[1] + reg_line[2]

    if total >= 0:
        if color == 'b':
            true_pos += 1
        else:
            false_neg += 1
    else:
        if color == 'r':
            true_neg += 1
        else:
            false_pos += 1

# print numbers of each rate
print(
    f"TRUE POSITIVE ENTRIES: {true_pos}\nTRUE NEGATIVE ENTRIES: {true_neg}\nFALSE POSITIVE ENTRIES: {false_pos}\nFALSE NEGATIVE ENTRIES: {false_neg}")

# calculate the different rates
accuracy = float(true_pos + true_neg)/float(true_pos +
                                            true_neg + false_neg + false_pos)
error = 1.0 - accuracy
true_pos_rate = float(true_pos)/float(true_pos + false_pos)
true_neg_rate = float(true_neg)/float(true_neg + false_neg)
false_pos_rate = float(false_pos)/float(true_pos + false_pos)
false_neg_rate = float(false_neg)/float(true_neg + false_neg)

# print rates
print(f"ACCURACY RATE: {accuracy*100.0}%\nERROR RATE: {error*100.0}%\nTRUE POSITIVE RATE: {true_pos_rate*100.0}%\nTRUE NEGATIVE RATE: {true_neg_rate*100.0}%\nFALSE POSITIVE RATE: {false_pos_rate *100.0}%\nFALSE NEGATIVE RATE: {false_neg_rate*100.0}%")


# scatter the plots using normalized coordinate lists, as well as colors. 
# Alpha is measure of opacity
opacity = 0.6
#  If 75% of data was trained
if trainingChoice == 1:
    if plotChoice == 1:
        ax.scatter(high_big_x_list, high_big_y_list, c = high_big_c_list, alpha = opacity)
        ax.scatter(high_small_x_list, high_small_y_list, c = high_small_c_list, alpha = opacity)
    elif plotChoice == 2:
        ax.scatter(low_big_x_list, low_big_y_list, c = low_big_c_list, alpha = opacity)
        ax.scatter(low_small_x_list, low_small_y_list, c = low_small_c_list, alpha = opacity)
# If 25% of data was trained
elif trainingChoice == 2:
    if plotChoice == 1:
        ax.scatter(low_big_x_list, low_big_y_list, c = low_big_c_list, alpha = opacity)
        ax.scatter(low_small_x_list, low_small_y_list, c = low_small_c_list, alpha = opacity)
    elif plotChoice == 2:
        ax.scatter(high_big_x_list, high_big_y_list, c = high_big_c_list, alpha = opacity)
        ax.scatter(high_small_x_list, high_small_y_list, c = high_small_c_list, alpha = opacity)

# configure and show plot
ax.set_title(f'Scatter plot of {group_name} with normalized weight vs cost')
ax.set_xlabel('Weight')
ax.set_ylabel('Cost')
show()
