import numpy as np


def calculate_index(x: int, y: int, w: int) -> (int, int):
    index1 = [x * w + y, (x + 1) * w + y, (x + 1) * w + y + 1]
    index2 = [x * w + y, x * w + y + 1, (x + 1) * w + y + 1]
    return index1, index2


def generate_vertices(width: int, height: int):
    vertices = []
    indices = []
    for i in range(width):
        for j in range(height):
            vertices.append(i * 0.5)
            vertices.append(j * 0.5)

    for i in range(width - 1):
        for j in range(height):
            index1, index2 = calculate_index(i, j, width)
            indices.extend(index1)
            indices.extend(index2)
    return np.array(vertices), np.array(indices)

