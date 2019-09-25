import numpy as np


def calculate_index(x: int, y: int, w: int) -> (int, int):
    index1 = [x * w + y, (x + 1) * w + y, (x + 1) * w + y + 1]
    index2 = [x * w + y, x * w + y + 1, (x + 1) * w + y + 1]
    return index1, index2


def generate_vertices(start_position: (float, float), shape: (int, int), x_distance: float, y_distance: float):
    vertices = []
    indices = []
    for i in range(shape[0]):
        for j in range(shape[1]):
            vertices.append(start_position[0] + i * x_distance)
            vertices.append(start_position[1] + j * y_distance)

    for i in range(shape[0] - 1):
        for j in range(shape[1] - 1):
            index1, index2 = calculate_index(i, j, shape[1])
            indices.extend(index1)
            indices.extend(index2)
    return np.array(vertices), np.array(indices)

