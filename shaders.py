
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