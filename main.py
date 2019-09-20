import struct
import sys

import noise
import numpy as np
import pygame
from OpenGL.GLU import *
from opensimplex import OpenSimplex
from pygame.locals import *

from buffer_objects import *
from shaders import *
from shaders_src import *


class TerrainAnimationGLData:
    def __init__(self):
        self.vbo: VBO = None
        self.ibo: IBO = None
        self.height_bo: BO = BO()
        self.shader: Shader = None
        self.shader2: Shader = None
        self.vertices = None
        self.indices = None

    def initialize(self):
        self.shader = Shader()
        self.shader.compile(vertex_shader, fragment_shader)
        self.shader2 = Shader()
        self.shader2.compile(vertex_shader, fragment_shader2)
        self.vbo = VBO()
        self.ibo = IBO()
        self.vbo.set_vertex_attribute(2, len(self.vertices) * sys.getsizeof(float()),
                                      struct.pack(str(len(self.vertices)) + "f", *self.vertices))
        self.ibo.set_indices(4, len(self.indices) * sys.getsizeof(int()), struct.pack(str(len(self.indices)) + "I",
                                                                                      *self.indices))

    def draw(self):
        if not self.vbo:
            self.initialize()

        # mvp_matrix_id, mvp = create_mvp(self.shader.program, 800, 600)
        # glUniformMatrix4fv(mvp_matrix_id, 1, GL_FALSE, mvp)

        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        self.shader.use()
        self.vbo.set_slot(0)
        self.ibo.bind()
        self.height_bo.set_slot(1)
        self.ibo.draw()

        self.ibo.unbind()
        self.vbo.unbind()
        self.height_bo.unbind()
        self.shader.unuse()

        '''glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        self.shader2.use()
        self.vbo.set_slot(0)
        self.ibo.bind()
        self.height_bo.set_slot(1)
        self.ibo.draw()

        self.ibo.unbind()
        self.vbo.unbind()
        self.height_bo.unbind()
        self.shader2.unuse()'''


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

    return result


def apply_map_to_vertices(map_config: GenerationConfig, map: GenerationResult, verticies):
    for i in range(map_config.shape[0]):
        for j in range(map_config.shape[1]):
            verticies[i * map_config.shape[0]][1] = map.data[i][j]


def draw(vertices, indices):
    glBegin(GL_TRIANGLES)
    for index in indices:
        for vertex in index:
            glVertex2fv(vertices[vertex])
    glEnd()


def perspective(fov, aspect, near, far):
    n, f = near, far
    t = np.tan((fov * np.pi / 180) / 2) * near
    b = - t
    r = t * aspect
    l = b * aspect
    assert abs(n - f) > 0
    return np.array((
        ((2 * n) / (r - l), 0, 0, 0),
        (0, (2 * n) / (t - b), 0, 0),
        ((r + l) / (r - l), (t + b) / (t - b), (f + n) / (n - f), -1),
        (0, 0, 2 * f * n / (n - f), 0)))


def normalized(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def look_at(eye, target, up):
    zax = normalized(eye - target)
    xax = normalized(np.cross(up, zax))
    yax = np.cross(zax, xax)
    x = - xax.dot(eye)
    y = - yax.dot(eye)
    z = - zax.dot(eye)
    return np.array(((xax[0], yax[0], zax[0], 0),
                     (xax[1], yax[1], zax[1], 0),
                     (xax[2], yax[2], zax[2], 0),
                     (x, y, z, 1)))


def create_mvp(program_id, width, height):
    fov, near, far = 70, 0.001, 100
    eye = np.array((10, 2, 15))
    target, up = np.array((10, 0, 10)), np.array((0, 1, 0))
    projection = perspective(fov, width / height, near, far)
    view = look_at(eye, target, up)
    model = np.identity(4)
    mvp = model @ view @ projection
    print(mvp)
    matrix_id = glGetUniformLocation(program_id, 'mvp')
    return matrix_id, mvp.astype(np.float32)


x_offset = 0
speed = 0.5
create_mvp(0, 800, 600)


def main():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)

    glTranslatef(0.0, 0.0, -5)

    map_config = GenerationConfig()
    vertices, indices = generate_vertices(map_config)
    data = TerrainAnimationGLData()
    data.vertices = vertices
    data.indices = indices

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_LINE_SMOOTH)
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
    glCullFace(GL_FRONT)

    def disp_func():
        global x_offset
        global speed
        speed += 0.005

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        map_config.amplitude_scale = 10
        map_config.simplex_step = 5
        map = generate_map(map_config, x_offset, 'perlin')
        data.height_bo.set_data(1, len(map.data) * sys.getsizeof(float()),
                                struct.pack(str(len(map.data)) + "f", *map.data))
        data.draw()
        glFlush()
        x_offset -= speed

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        glRotatef(1, 1, 1, 1)
        disp_func()
        pygame.display.flip()
        pygame.time.wait(1)


main()
