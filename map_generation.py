from opensimplex import OpenSimplex
import noise
import numpy as np


class GenerationConfig:
    def __init__(self):
        self.shape = (30, 30)
        self.scale = 100.0
        self.octaves = 6
        self.persistence = 0.5
        self.lacunarity = 2.0
        self.amplitude_scale = 1.0

        self.simplex_step = 1.0


class GenerationResult:
    def __init__(self, data):
        self.data = data


def generate_map(map_config: GenerationConfig, x_offset: float, map_type: str) -> GenerationResult:
    if map_type == 'perlin':
        data = np.zeros(map_config.shape, dtype=np.float32)
        for i in range(map_config.shape[0]):
            for j in range(map_config.shape[1]):
                data[i][j] = noise.pnoise2(i / map_config.scale,
                                           (j + x_offset) / map_config.scale,
                                           octaves=map_config.octaves,
                                           persistence=map_config.persistence,
                                           lacunarity=map_config.lacunarity,
                                           repeatx=10,
                                           repeaty=10,
                                           base=0) * map_config.amplitude_scale
        result = GenerationResult(data.flatten())
    elif map_type == 'simplex':
        gen = OpenSimplex(seed=1)
        data = np.zeros(map_config.shape, dtype=np.float32)
        for i in range(map_config.shape[0]):
            for j in range(map_config.shape[1]):
                data[i][j] = gen.noise2d(i / map_config.scale * map_config.simplex_step,
                                         (j + x_offset) / map_config.scale * map_config.simplex_step) \
                             * map_config.amplitude_scale
        result = GenerationResult(data.flatten())
    else:
        raise Exception('Invalid type:' + map_type)
    return result
