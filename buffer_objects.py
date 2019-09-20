from OpenGL.GL import *
from typing import Any


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

    def set_vertex_attribute(self, component_count: int, byte_length: int, data: any):
        self.component_count = component_count
        stride = 4 * self.component_count
        self.vertex_count = byte_length // stride
        self.bind()
        glBufferData(GL_ARRAY_BUFFER, byte_length, data, GL_STATIC_DRAW)

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

    def set_indices(self, stride: int, byte_length: int, data: Any):
        self.index_count = byte_length // stride
        self.bind()
        if stride == 1:
            self.index_type = GL_UNSIGNED_BYTE
        elif stride == 2:
            self.index_type = GL_UNSIGNED_SHORT
        elif stride == 4:
            self.index_type = GL_UNSIGNED_INT
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, byte_length, data, GL_STATIC_DRAW)

    def draw(self):
        glDrawElements(GL_TRIANGLES, self.index_count, self.index_type, None)
