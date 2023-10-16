import matplotlib.pyplot as plt
import numpy as np

my_list = []

with open("./Project1_data-1/groupA.txt") as file:
    prices = []
    weights = []
    binaryWeights = []
    for line in file:
        # print(line.rstrip().split(','))
        vechicle = line.rstrip().split(',')
        price = float(vechicle[0])
        weight = float(vechicle[1])
        isHeavy = float(vechicle[2])
        prices.append(price)
        weights.append(weight)
        binaryWeights.append(isHeavy)

    plt.scatter(prices, weights)
    plt.show()                   # Display the plot
