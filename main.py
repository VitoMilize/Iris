import seaborn as snc
import numpy as np


def dist(point1, point2):
    distance = 0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return distance ** 0.5


def take_k_nearest_objects(k, array):
    return array[:k, :]


dataset = snc.load_dataset("iris")

features = dataset.drop('species', axis=1).values
targets = dataset['species'].values

sepal_length_input = float(input("Введите sepal_length: "))
sepal_width_input = float(input("Введите sepal_width: "))
petal_length_input = float(input("Введите petal_length: "))
petal_width_input = float(input("Введите petal_width: "))

params = np.array([sepal_length_input, sepal_width_input, petal_length_input, petal_width_input])

distances = np.array([dist(params, feature) for feature in features])
distances_with_indexes = np.column_stack((np.arange(len(distances)), distances))
sorted_indexes = np.argsort(distances_with_indexes[:, 1])
sorted_distances_with_indexes = distances_with_indexes[sorted_indexes]

nearest_objects = take_k_nearest_objects(5, sorted_distances_with_indexes)

counter = dict()
for i in range(len(nearest_objects)):
    target_value = targets[int(nearest_objects[i, 0])]
    if target_value in counter:
        counter[target_value] += 1
    else:
        counter[target_value] = 1

most_common_value = max(counter, key=counter.get)

print(most_common_value)
