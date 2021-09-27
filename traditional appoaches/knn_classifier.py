import math
import statistics
from collections import Counter

def euclidean_distance(p1, p2):
    distance = 0
    for i in range(len(p1)):
        distance += math.pow(p1[i] - p2[i], 2)
    return distance

# for regression task
def mean(sample):
    return sum(sample) / len(sample)

# for classification task 
def mode(sample):
    c = Counter(sample)
    return c.most_common(1)[0][0]

def k_nearest_neighbor(k, data, query, task_type):
    ordered = []
    for i, val in enumerate(data):
        distance = euclidean_distance(val[:-1], query)
        ordered.append((distance, i))
    
    k_neighbor = sorted(ordered)[:k]
    k_labels = [entry[i][-1] for distance, index in k_neighbor]
    result = task_type(k_labels)
    return result

def main():

    classification_data = [
       [22, 1],
       [23, 1],
       [21, 1],
       [18, 1],
       [19, 1],
       [25, 0],
       [27, 0],
       [29, 0],
       [31, 0],
       [45, 0],
    ]
    # Question:
    # Given the data we have, does a 33 year old like pineapples on their pizza?
    query = [33]

    classification = k_nearest_neighbor(
        3, classification_data, query, mode
    )

    if __name__ == '__main__':
        main() 
