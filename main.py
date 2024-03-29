import struct
import sys

import pygame
from pygame.locals import *

from buffer_objects import *
from shaders import *
from shaders_src import *
from vertices_indices import *
from map_generation import *
from mvp import *

from xmlrpc.server import SimpleXMLRPCServer, SimpleXMLRPCRequestHandler
import threading


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


class AnimationSettings:
    def __init__(self):
        self.speed = 0.1
        self.z_offset = 0
        self.map_config = GenerationConfig()
        self.generator_type = 'perlin'

    def step_position(self):
        self.z_offset -= self.speed

    def set_generator_tyoe(self, type: str):
        self.generator_type = type
        return self.generator_type

    def add_to_speed(self, speed: float):
        self.speed += speed
        return self.speed

    def add_to_scale(self, scale: float):
        self.map_config.scale += scale
        return self.map_config.scale

    def add_to_octaves(self, octaves: float):
        self.map_config.octaves += octaves
        return self.map_config.octaves

    def add_to_persistence(self, persistence: float):
        self.map_config.persistence += persistence
        return self.map_config.persistence

    def add_to_lacunarity(self, lacunarity: float):
        self.map_config.lacunarity += lacunarity
        return self.map_config.lacunarity

    def add_to_amplitude_scale(self, amplitude_scale: float):
        self.map_config.amplitude_scale += amplitude_scale
        return self.map_config.amplitude_scale

    def add_to_simplex_step(self, simplex_step: float):
        self.map_config.simplex_step += simplex_step
        return self.map_config.simplex_step


class RPCController:
    def __init__(self, animation_settings: AnimationSettings):
        self.animation_settings = animation_settings

    def run(self):
        def serve():
            print('Running RPC on localhost:8000')
            with SimpleXMLRPCServer(('localhost', 8000), requestHandler=SimpleXMLRPCRequestHandler) as server:
                server.register_introspection_functions()

                server.register_instance(self.animation_settings)
                server.serve_forever()
        self.thread = threading.Thread(target=serve)
        self.thread.start()


def setup_opengl():
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_LINE_SMOOTH)
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
    glCullFace(GL_FRONT)


print(create_mvp(800, 600))

def main():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    # pygame.display.toggle_fullscreen()

    settings = AnimationSettings()
    settings.map_config.shape = (40, 40)
    settings.map_config.amplitude_scale = 10
    settings.map_config.simplex_step = 5
    settings.speed = 0.1
    settings.generator_type = 'perlin'

    controller = RPCController(settings)
    controller.run()

    vertices, indices = generate_vertices((-5, -5), settings.map_config.shape, 1, 1)
    data = TerrainAnimationGLData()
    data.vertices = vertices
    data.indices = indices

    setup_opengl()

    def display():
        glClearColor(0.75, 0.129, 0.357, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        map = generate_map(settings.map_config, settings.z_offset, settings.generator_type)
        data.height_bo.set_data(1, len(map.data) * sys.getsizeof(float()),
                                struct.pack(str(len(map.data)) + "f", *map.data))
        data.draw()
        glFlush()
        settings.step_position()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        display()
        pygame.display.flip()
        pygame.time.wait(1)


main()
