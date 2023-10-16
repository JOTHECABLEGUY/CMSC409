from matplotlib.pyplot import subplots, show
import random
import os
import math

weights, costs, colors, normalized_costs, normalized_weights = [],[],[],[],[]				# initialize lists														# init list vars
files = [f'{os.getcwd()}/{f}' for f in os.listdir(os.getcwd()) if f.endswith(".txt")]		# find text files
fig, ax = subplots()																		# get subplots
fileDict = dict()																			# init dict to print files

# build filename dictionary
for i in range(0, len(files)):
	fileDict[i+1] = files[i]

# print the dictionary for user to see
for key, value in fileDict.items():
	val = value.split("/")[-1]
	print(f"{key}: {val}")

# store input from user to use as a key
x = int(input("Enter the NUMBER corresponding to the file you want to open:  "))

# read file
with open(fileDict[x]) as f:
	contents = f.readlines()
	group_name = f.name.split('/')[-1][:-4] 						# format name of group

# use contents of text file to build coordinate lists
for line in contents:
	line = line.strip().split(",") 		# format each line from file
	color = 'r' if line[2] == '0' else 'b'	# configure color based on if 'small' or 'big'
	weight = float(line[1])
	cost = float(line[0])
	weights.append(weight)
	costs.append(cost)
	colors.append(color)

# set maxima and minima
maxCost = max(costs)
minCost = min(costs)
maxWeight = max(weights)
minWeight = min(weights)

# function to normalize data
def normalize(x, minimum, maximum):
	numerator = x - minimum
	denominator = maximum - minimum
	return float(numerator/denominator)

big_list = []
small_list = []
# normalize both weight and cost
for weight, cost, color in zip(weights, costs, colors):
	normalized_weight = normalize(weight, minWeight, maxWeight)
	normalized_weights.append(normalized_weight)
	normalized_cost = normalize(cost, minCost, maxCost)
	normalized_costs.append(normalized_cost)
	if color == 'b':
		big_list.append([normalized_weight, normalized_cost, color])
	else:
		small_list.append([normalized_weight, normalized_cost, color])



high_big_x_list = list()
high_big_y_list = list()
high_big_c_list = list()
low_big_x_list = list()
low_big_y_list = list()
low_big_c_list = list()

high_small_x_list = list()
high_small_y_list = list()
high_small_c_list = list()
low_small_x_list = list()
low_small_y_list = list()
low_small_c_list = list()

ws = [random.random()-0.5, random.random()-0.5, -0.5]
print(ws)

for i in range(len(big_list)):
	if not (i%4 == 0):
		high_big_x_list.append(big_list[i][0])
		high_big_y_list.append(big_list[i][1])
		high_big_c_list.append(big_list[i][2])
	else:
		low_big_x_list.append(big_list[i][0])
		low_big_y_list.append(big_list[i][1])
		low_big_c_list.append(big_list[i][2])

for i in range(len(small_list)):
	if not (i%4 == 0):
		high_small_x_list.append(small_list[i][0])
		high_small_y_list.append(small_list[i][1])
		high_small_c_list.append(small_list[i][2])
	else:
		low_small_x_list.append(small_list[i][0])
		low_small_y_list.append(small_list[i][1])
		low_small_c_list.append(small_list[i][2])
patterns = [[x, y, 1] for x,y in zip(high_big_x_list, high_big_y_list)] + [[x, y, 1] for x,y in zip(high_small_x_list, high_small_y_list)]
d_out = [1]*int((len(patterns)/2)) + [-1]*int((len(patterns)/2))
print(d_out)
# small_patterns = 
# patterns = big_patterns+small_patterns
# centers = [[(max(high_big_x_list)+min(high_big_x_list))/2, (max(high_big_y_list)+min(high_big_y_list))/2, 1], 
# 	[(max(high_small_x_list)+min(high_small_x_list))/2, (max(high_small_y_list)+min(high_small_y_list))/2, 1]]
print(len(patterns))

def sign(net):
	if net >= 0:
		return 1
	else:
		return -1


# def fbip(net):
# 	k = 0.2
# 	return 2 / (1 + math.exp(-2*k*net))


# def printdata(iteration, pattern, net, err, learn, ww):
# 	# itc = 0  p = 0 net = 4.00 err = 2.0 lrn = 0.200
# 	# weights: 1.00 3.00 -3.00
# 	ww_formatted = ['%.5f' % elem for elem in ww]
# 	print("ite= ", iteration, ' pat =', pattern, ' net=', round(net, 5),
# 		  ' err=', err, ' lrn=', learn, ' weights=', ww_formatted)

# def soft_percept(ite, num_patterns, num_dimen, alpha, ws, patterns, desired_out):
# 	# ite = 8  # number of training cycles
# 	# np = 2  # number of patterns
# 	# ni = 3  # number of augmented inputs
# 	# alpha = 0.1  # learning constant
# 	# ww = [1, 3, -3]  # array of weights
# 	# pat = ([1, 2, 1], [2, 1, 1])  # patterns as 2-dim array
# 	# dout = (-1, 1)  # desired output as 1-dim array
# 	# # print(type(ww))

# 	for iteration in range(0, ite):  # number of training cycles
# 		ou = [0, 0]  # temporary array to store the output
# 		for pattern in range(0, num_patterns):  # for all patterns
# 			net = 0
# 			for i in range(0, num_dimen):  # for all inputs
# 				net = net + ws[i]*patterns[pattern][i]

# 			ou[pattern] = fbip(net)  # use activation function
# 			if ou[pattern] < math.pow(10,-5):
# 				return ws
# 			# print(f"out = {ou[pattern]}")
# 			err = desired_out[pattern] - ou[pattern]
# 			learn = alpha * err
# 			# if pattern % num_dimen == 0:
# 			printdata(iteration, pattern, net, err, learn, ws)
# 			for i in range(0, num_dimen):	# for all inputs
# 			#print(wqw[i] + 1)
# 				ws[i] = ws[i] + learn * patterns[pattern][i]
# 	return ws

def net_hard(wws, patt):
	net = 0
	for w, p in zip(wws, patt):
		net += w*p
	return net

def delta_hard(a, x, d, o):
	delta = [0]*len(x)
	# print(f'a = {a}\nd = {d}\no = {o}')
	coeff = a*(d-o)
	# print(coeff)
	for i in range(len(x)):
		delta[i] = x[i]*coeff
	return delta

def hard_percept(ite, alpha, patterns, ws, desired_out):
	for iteration in range(ite):
		for p, d in zip(patterns, desired_out):
			# print(f'P ==> D :: {p} ==> {d}')
			net = net_hard(ws, p)
			# print(f'net = {net}, sign of net = {sign(net)}')
			delta_w = delta_hard(alpha, p, d, sign(net))
			for i in range(len(ws)):
				ws[i] += delta_w[i]
			print("ite= ", iteration, ' pat =', p, ' net=', round(net, 5),' weights=', ws, ' change=', delta_w)
	return ws

new_weights = hard_percept(1000, 0.2, patterns, ws, d_out)
print(new_weights)

# scatter the plots using normalized coordinate lists, as well as colors. Alpha is measure of opacity
ax.scatter(high_big_x_list, high_big_y_list, c = high_big_c_list, alpha = 0.6)
ax.scatter(high_small_x_list, high_small_y_list, c = high_small_c_list, alpha = 0.6)

# classification line
y_0 = new_weights[2]/new_weights[1]
x_0 = new_weights[2]/new_weights[0]
ax.plot([abs(y_0), 0], [0, abs(x_0)])
reg_line = new_weights

# init confusion vars
true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0

# run classification line
for weight, cost, color in zip(high_big_x_list, high_big_y_list, ['b']*len(high_big_y_list)):
	total = weight * reg_line[0] + cost * reg_line[1] + reg_line[2]
	
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

for weight, cost, color in zip(high_small_x_list, high_small_y_list, ['r']*len(high_small_x_list)):
	total = weight * reg_line[0] + cost * reg_line[1] + reg_line[2]
	
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
print(f"TRUE POSITIVE ENTRIES: {true_pos}\nTRUE NEGATIVE ENTRIES: {true_neg}\nFALSE POSITIVE ENTRIES: {false_pos}\nFALSE NEGATIVE ENTRIES: {false_neg}")

# calculate the different rates
accuracy = float(true_pos + true_neg)/float(true_pos + true_neg + false_neg + false_pos)
error = 1.0 - accuracy
true_pos_rate = float(true_pos)/float(true_pos + false_pos)
true_neg_rate = float(true_neg)/float(true_neg + false_neg)
false_pos_rate = float(false_pos)/float(true_pos + false_pos)
false_neg_rate = float(false_neg)/float(true_neg + false_neg)

# print rates
print(f"ACCURACY RATE: {accuracy*100.0}%\nERROR RATE: {error*100.0}%\nTRUE POSITIVE RATE: {true_pos_rate*100.0}%\nTRUE NEGATIVE RATE: {true_neg_rate*100.0}%\nFALSE POSITIVE RATE: {false_pos_rate *100.0}%\nFALSE NEGATIVE RATE: {false_neg_rate*100.0}%")

# configure and show plot
ax.set_title(f'Scatter plot of {group_name} with normalized weight vs cost')
ax.set_xlabel('weight')
ax.set_ylabel('cost')
show()
