import pygame
import pyrr as pyrr
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import noise
import numpy as np
import sys
import struct
from typing import Any
from pyrr import Matrix44, Vector4, Vector3, Quaternion
from opensimplex import OpenSimplex

vertex_shader = '''
#version 120
attribute vec2 position;
attribute float height;

varying float amplitude;

//uniform mat4 mvp;

void main() {
    mat4 mvp = mat4(  1.07111101,   0.0,           0.0,           0.0,        
  0.0,           1.32600214,  -0.3713981,   -0.37139068,
  0.0,          -0.53040085,  -0.92849526,  -0.92847669,
-10.71111005,   5.30400854,  14.6682251,   14.66993172);
    gl_Position = mvp * vec4(position[0], height, position[1], 1);
    amplitude = height;
}
'''

fragment_shader = '''
#version 120

varying float amplitude;

void main() {
    float tmp = amplitude;
    vec3 color = vec3(1, 0.5, 0.5);
    if (tmp > 1.0) {
        tmp = 1.0;
    } else if (tmp < 0.0) {
        color = vec3(0.5, 0.5, 1);
        tmp = abs(tmp);
        if (tmp > 1.0) {
            tmp = 1.0;
        }
    }
    color = color * tmp;
    gl_FragColor = vec4(color, 1);
}
'''

fragment_shader2 = '''
#version 120

varying float amplitude;

void main() {
    float tmp = amplitude;
    vec3 color = vec3(1, 0.5, 0.5);
    if (tmp > 1.0) {
        tmp = 1.0;
    } else if (tmp < 0.0) {
        color = vec3(0.5, 0.5, 1);
        tmp = abs(tmp);
        if (tmp > 1.0) {
            tmp = 1.0;
        }
    }
    color = color * tmp;
    gl_FragColor = vec4(color, tmp);
}
'''

def load_shader(src: str, shader_type: int) -> int:
    shader = glCreateShader(shader_type)
    glShaderSource(shader, src)
    glCompileShader(shader)
    error = glGetShaderiv(shader, GL_COMPILE_STATUS)
    if error != GL_TRUE:
        info = glGetShaderInfoLog(shader)
        glDeleteShader(shader)
        raise Exception(info)
    return shader


class Shader:
    def __init__(self):
        self.program = glCreateProgram()

    def __del__(self):
        glDeleteProgram(self.program)

    def compile(self, vs_src: str, fs_src: str):
        vs = load_shader(vs_src, GL_VERTEX_SHADER)
        if not vs:
            return
        fs = load_shader(fs_src, GL_FRAGMENT_SHADER)
        if not fs:
            return
        glAttachShader(self.program, vs)
        glAttachShader(self.program, fs)
        glLinkProgram(self.program)
        error = glGetProgramiv(self.program, GL_LINK_STATUS)
        glDeleteShader(vs)
        glDeleteShader(fs)
        if error != GL_TRUE:
            info = glGetShaderInfoLog(self.program)
            raise Exception(info)

    def use(self):
        glUseProgram(self.program)

    def unuse(self):
        glUseProgram(0)


class VBO:
    def __init__(self):
        self.vbo = glGenBuffers(1)
        self.component_count = 0
        self.vertex_count = 0

    def __del__(self):
        glDeleteBuffers(1, [self.vbo])

    def bind(self):
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)

    def unbind(self):
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def set_vertex_attribute(self, component_count: int, bytelength: int,
                             data: any):
        self.component_count = component_count
        stride = 4 * self.component_count
        self.vertex_count = bytelength // stride
        self.bind()
        glBufferData(GL_ARRAY_BUFFER, bytelength, data, GL_STATIC_DRAW)

    def set_slot(self, slot: int):
        self.bind()
        glEnableVertexAttribArray(slot)
        glVertexAttribPointer(slot, self.component_count, GL_FLOAT, GL_FALSE, 0, None)

    def draw(self):
        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)


class BO:
    def __init__(self):
        self.bo = glGenBuffers(1)
        self.component_count = 0

    def __del__(self):
        glDeleteBuffers(1, [self.bo])

    def bind(self):
        glBindBuffer(GL_ARRAY_BUFFER, self.bo)

    def unbind(self):
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def set_data(self, component_count: int, byte_length: int, data: any):
        self.component_count = component_count
        self.bind()
        glBufferData(GL_ARRAY_BUFFER, byte_length, data, GL_DYNAMIC_DRAW)

    def set_slot(self, slot: int):
        self.bind()
        glEnableVertexAttribArray(slot)
        glVertexAttribPointer(slot, self.component_count, GL_FLOAT, GL_FALSE, 0, None)


class IBO:
    def __init__(self):
        self.vbo = glGenBuffers(1)
        self.index_count = 0
        self.index_type = 0

    def __del__(self):
        glDeleteBuffers(1, [self.vbo])

    def bind(self):
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vbo)

    def unbind(self):
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

    def set_indices(self, stride: int, bytelength: int, data: Any):
        self.index_count = bytelength // stride
        self.bind()
        if stride == 1:
            self.index_type = GL_UNSIGNED_BYTE
        elif stride == 2:
            self.index_type = GL_UNSIGNED_SHORT
        elif stride == 4:
            self.index_type = GL_UNSIGNED_INT
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, bytelength, data, GL_STATIC_DRAW)

    def draw(self):
        glDrawElements(GL_TRIANGLES, self.index_count, self.index_type, None)


class Data:
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

class MapConfig:
    def __init__(self):
        self.shape = (30, 30)
        self.scale = 100.0
        self.octaves = 6
        self.persistence = 0.5
        self.lacunarity = 2.0
        self.amplitude_scale = 1.0

        self.simplex_step = 1.0


class Map:
    def __init__(self, data):
        self.data = data


def calculate_index(x: int, y: int, w: int) -> (int, int):
    index1 = [x * w + y, (x + 1) * w + y, (x + 1) * w + y + 1]
    index2 = [x * w + y, x * w + y + 1, (x + 1) * w + y + 1]
    return index1, index2


def generate_vertices(map_config: MapConfig):
    vertices = []
    indices = []
    for i in range(map_config.shape[0]):
        for j in range(map_config.shape[1]):
            vertices.append(i * 0.5)
            vertices.append(j * 0.5)

    for i in range(map_config.shape[0] - 1):
        for j in range(map_config.shape[1] - 1):
            index1, index2 = calculate_index(i, j, map_config.shape[0])
            indices.extend(index1)
            indices.extend(index2)
    return np.array(vertices), np.array(indices)


def generate_map(map_config: MapConfig, x_offset: float, map_type: str) -> Map:
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
        result = Map(data.flatten())
    elif map_type == 'simplex':
        gen = OpenSimplex(seed=1)
        data = np.zeros(map_config.shape, dtype=np.float32)
        for i in range(map_config.shape[0]):
            for j in range(map_config.shape[1]):
                data[i][j] = gen.noise2d(i / map_config.scale * map_config.simplex_step,
                                         (j + x_offset) / map_config.scale * map_config.simplex_step) \
                             * map_config.amplitude_scale
        result = Map(data.flatten())

    return result


def apply_map_to_vertices(map_config: MapConfig, map: Map, verticies):
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

    map_config = MapConfig()
    vertices, indices = generate_vertices(map_config)
    data = Data()
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
