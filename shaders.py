from OpenGL.GL import *


def load_shader(src: str, shader_type: int) -> int:
    shader = glCreateShader(shader_type)
    glShaderSource(shader, src)
    glCompileShader(shader)
    gl_error = glGetShaderiv(shader, GL_COMPILE_STATUS)
    if gl_error != GL_TRUE:
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
        fs = load_shader(fs_src, GL_FRAGMENT_SHADER)

        glAttachShader(self.program, vs)
        glAttachShader(self.program, fs)
        glLinkProgram(self.program)
        gl_error = glGetProgramiv(self.program, GL_LINK_STATUS)
        glDeleteShader(vs)
        glDeleteShader(fs)
        if gl_error != GL_TRUE:
            info = glGetShaderInfoLog(self.program)
            raise Exception(info)

    def use(self):
        glUseProgram(self.program)

    def unuse(self):
        glUseProgram(0)
