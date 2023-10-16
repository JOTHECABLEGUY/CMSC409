from matplotlib.pyplot import subplots, show
import os

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

# normalize both weight and cost
for weight, cost in zip(weights, costs):
	normalized_weight = normalize(weight, minWeight, maxWeight)
	normalized_weights.append(normalized_weight)
	normalized_cost = normalize(cost, minCost, maxCost)
	normalized_costs.append(normalized_cost)


# scatter the plots using normalized coordinate lists, as well as colors. Alpha is measure of opacity
ax.scatter(normalized_weights, normalized_costs, c = colors, alpha = 0.6)

# classification line
ax.plot([0, 1], [1, 0])  
#reg_line = (1, 1, -.989)
reg_line = (1,1,-1)

# init confusion vars
true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0

# run classification line
for weight, cost, color in zip(normalized_weights, normalized_costs, colors):
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
