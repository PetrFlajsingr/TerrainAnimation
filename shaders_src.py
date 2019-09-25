
vertex_shader = '''
#version 120
attribute vec2 position;
attribute float height;

varying float amplitude;

//uniform mat4 mvp;

void main() {
    float h = height;
    if (height < 0.0) {
        h = 0.0;
    } 
    mat4 mvp = mat4(  1.071111,     0.,           0.,           0.,        
  0.,           1.0098531,   -0.7071209,   -0.70710677,
  0.,          -1.0098531,   -0.7071209,   -0.70710677,
-10.71111,     10.098532,    14.140418,    14.142136);
    gl_Position = mvp * vec4(position[0], h, position[1], 1);
    amplitude = h;
}
'''

fragment_shader = '''
#version 120

varying float amplitude;

void main() {
    float tmp = amplitude;
    vec3 color = vec3(0.514, 0.478, 0.941);
    if (tmp < 0.0) {
        color = vec3(0.5, 0.5, 1);
        tmp = abs(tmp);
        if (tmp > 1.0) {
            tmp = 1.0;
        }
        tmp = 1.0 - tmp;
    } else {
        tmp += 1.0;
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
    gl_FragColor = vec4(color, 1);
}
'''
