import pandas as pd 
import numpy as np 
import random 
import matplotlib.pyplot as plt 

class Point:
    def __init__(self, x, y, c):
        self.x = x
        self.y = y 
        self.color = c
        if self.color == 'navy':
            self.type = 1
        else:
            self.type = 0
    
    def __repr__(self):
        return f'({self.x}, {self.y})'

    def plot(self):
        plt.plot(self.x, self.y, color = self.color, marker = 'o')
    
points = []

# linear separable
# for _ in range(100):
#     x = random.randint(0, 20)
#     y = random.randint(0, 20)
#     if x + y < 9:
#         points.append(Point(x, y, 'fuchsia'))
#     elif x + y > 11:
#         points.append(Point(x, y, 'navy')) 

# not linear separable
for _ in range(50):
    x = random.randint(0, 18)
    y = random.randint(0, 18)
    x2 = random.randint(2, 20)
    y2 = random.randint(2, 20)
    points.append(Point(x, y, 'fuchsia'))

    points.append(Point(x2, y2, 'navy')) 

inputs = []
labels = []
for p in points:
    inputs.append([p.x, p.y])
    labels.append(p.type)
inputs = np.array(inputs)
labels = np.array(labels)
# print(inputs)
# print(labels)
# print(f'points: {points}')

########### PERCEPTRON ############
class Perceptron:
    
    def __init__(self, n_inputs, threshold = 100, learning_rate = 0.01):
        self.threshold = threshold 
        self.learning_rate = learning_rate
        self.weights = np.zeros(n_inputs+1) # to account for the first 1
    
    def predict(self, inputs):
        s = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if s > 0:
            return 1
        else:
            return 0
    
    def train(self, training_inputs, labels):
        # best_weights = []
        # count = 0
        best_misclassified = 10000
        for _ in range(10*self.threshold):
            # if (_ % 25) == 0:
            #     self.draw_line()
            misclassified = 0
            for inputs, label in zip(training_inputs, labels):
                # if count == 200:
                #     x1, y1 = [-1, 6], [2.1, 1]
                #     plt.xlim(-1, 6)
                #     plt.ylim(-1, 6)
                #     plt.plot(x1, y1, label = 'previous')
                    
                prediction = self.predict(inputs)
                if prediction != label:
                    misclassified += 1
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs 
                self.weights[0] += self.learning_rate * (label - prediction)
            # print(f'misclassified: {misclassified}')
                # if count == 201:
                #     x1, y1 = [-1, 5.2], [3, 0.5]
                #     plt.xlim(-1, 6)
                #     plt.ylim(-1, 6)
                #     plt.plot(x1, y1, label = 'next')
                
                # count += 1
            if misclassified < best_misclassified:
                best_misclassified = misclassified
                best_weights = self.weights
                percentage = misclassified/len(training_inputs) * 100

        self.weights = best_weights
        self.percentage = percentage 

    def draw_line(self):
        # plt.clf()
        x1, y1 = [0, -self.weights[0]/self.weights[1]], [-self.weights[0]/self.weights[2], 0]
        plt.xlim(0, 20)
        plt.ylim(0, 20)
        plt.plot(x1, y1)
        plt.savefig("not_linear.png")
        

perceptron = Perceptron(2)
perceptron.train(inputs, labels)
for w in perceptron.weights:
    print(round(w, 2))

# Plotting a line #
perceptron.draw_line()
print(f"with {perceptron.percentage}% of error")

# plt.xlim(-2, 20), plt.ylim(-2, 20)
# plt.plot(x1, y1)
for p in points:
    p.plot()


plt.show()
