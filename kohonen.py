import random as r
import math as m

def norm(patterns, normed_weights = None, winner = None):

	# if the input patterns is a list of lists, then iterate on each list.
	# for each pattern in the list, square each element, sum the squares, and then take the square root. The resulting value is 
	# is the denominator of the normalization equation. Divide each element in the current pattern by the denominator to get the 
	# actual normalized values. Store in the same format as the input.
	if type(patterns[0]) == list:
		return [[ele/m.sqrt(sum([pow(ele, 2) for ele in pattern])) for ele in pattern] for pattern in patterns] 

	# if the above return statement was not reached, then there is only 1 pattern to normalize.
	# divide each element of the pattern by the same denominator as the one mentioned above, while also updating the normalized 
	# weights array. Return the updated weights array to retain any changes. 
	normed_weights[winner] = [ele/m.sqrt(sum([pow(ele, 2) for ele in patterns])) for ele in patterns]
	return normed_weights

def net(pattern, weights):

	# iterate through every neuron, calculating each neuron's net value by multiplying the input pattern by the 
	#	weight vector of the current neuron. Each row of net value is then summed to get the total net value for the neuron
	return [sum([row[i] * pattern[i] for i in range(len(pattern))]) for row in weights]

def update(weight, pattern, alpha):

	# update the winning neuron's weight (the weight parameter) according to the formula below:
	# w_i + ax_i
	# w is weight vector, a is alpha, x is the input pattern (normalized at this point)
	# result is a list of updated weights for the winning neuron (not normalized)
	return [weight[index] + alpha * pattern[index] for index in range(len(weight))]

if __name__ == '__main__':

	# define the patterns
	patterns = [[5.9630, 0.7258],[4.1168, 2.9694],[1.8184, 6.0148],[6.2139, 2.4288],[6.1290, 1.3876],[1.0562, 5.8288],[4.3185, 2.3792],[2.6108, 5.4870],[1.5999, 4.1317],[1.1046, 4.1969]]
	
	# normalize the patterns along each vector
	normed_patterns = norm(patterns)

	# get number of neurons (potential clusters)
	number_of_neurons = r.randint(2,4)

	# get number of iterations
	iterations = 30

	# number of inputs to each neuron depends on the length of each input pattern
	number_of_input = len(patterns[0])

	# define learning constant
	alpha = 0.3

	# randomly determine weights and normalize them according to same process as the patterns
	weights = [[r.randint(1,6) + r.uniform(0, 1) for i in range(number_of_input)] for j in range(number_of_neurons)]
	normed_weights = norm(weights)

	# run over each pattern 30 times
	for iteration in range(iterations):

		# for each pattern, update the network
		for pattern in normed_patterns:

			# find the values of net for each neuron and store in a list
			net_arr = net(pattern, normed_weights)

			# the winning neuron is the one with the highest net value in the list
			winner = net_arr.index(max(net_arr))

			# update the winning neuron, keeping others the same, and normalize the updated weight
			normed_weights = norm(update(normed_weights[winner], pattern, alpha), normed_weights, winner)

	# print the final weights of the network
	print(normed_weights)
