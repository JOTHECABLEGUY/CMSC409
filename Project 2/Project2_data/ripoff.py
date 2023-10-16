import math


def sign(net):
    if net >= 0:
        return 1
    else:
        return -1


def fbip(net):
    k = 0.2
    return 2 / (1 + math.exp(-2*k*net))


def printdata(iteration, pattern, net, err, learn, ww):
    # itc = 0  p = 0 net = 4.00 err = 2.0 lrn = 0.200
    # weights: 1.00 3.00 -3.00
    ww_formatted = ['%.5f' % elem for elem in ww]
    print("ite= ", iteration, ' pat =', pattern, ' net=', round(net, 5),
          ' err=', err, ' lrn=', learn, ' weights=', ww_formatted)


ite = 8  # number of training cycles
np = 2  # number of patterns
ni = 3  # number of augmented inputs
alpha = 0.1  # learning constant
ww = [1, 3, -3]  # array of weights
pat = ([1, 2, 1], [2, 1, 1])  # patterns as 2-dim array
dout = (-1, 1)  # desired output as 1-dim array
# print(type(ww))

for iteration in range(0, ite):  # number of training cycles
    ou = [0, 0]  # temporary array to store the output
    for pattern in range(0, np):  # for all patterns
        net = 0
        for i in range(0, ni):  # for all inputs
            net = net + ww[i]*pat[pattern][i]

        ou[pattern] = fbip(net)  # use activation function
        err = dout[pattern] - ou[pattern]
        learn = alpha * err
        printdata(iteration, pattern, net, err, learn, ww)
        for i in range(0, ni):  # for all inputs
            #print(wqw[i] + 1)
            ww[i] = ww[i] + learn * pat[pattern][i]
