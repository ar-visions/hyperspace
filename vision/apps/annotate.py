import moderngl
import moderngl_window
from moderngl_window.conf import settings
from pyrr import Matrix44, Vector3
import numpy as np

class CubeApp:
    def __init__(self):
        self.width, self.height = 800, 600

        settings.WINDOW['class'] = 'moderngl_window.context.glfw.Window'
        settings.WINDOW['gl_version'] = (4, 1)
        settings.WINDOW['size'] = (800, 600)

        self.window = moderngl_window.create_window_from_settings()

        self.ctx = moderngl.create_context()
        
        self.prog = self.ctx.program(
            vertex_shader='''
            #version 330
            uniform mat4 model;
            uniform mat4 projection;
            in vec3 in_position;
            out vec3 v_position;
            void main() {
                gl_Position = projection * model * vec4(in_position, 1.0);
                v_position = in_position;
            }
            ''',
            fragment_shader='''
            #version 330
            in vec3 v_position;
            out vec4 outColor;
            void main() {
                outColor = vec4((v_position + 1.0) / 2.0, 1.0);
            }
            '''
        )

        vertices = np.array([
            -0.5, -0.5, -0.5,   0.5, -0.5, -0.5,   0.5,  0.5, -0.5,  -0.5,  0.5, -0.5,  # Back face
            -0.5, -0.5,  0.5,   0.5, -0.5,  0.5,   0.5,  0.5,  0.5,  -0.5,  0.5,  0.5   # Front face
        ])
        
        indices = np.array([
            0, 1, 2, 2, 3, 0,
            4, 5, 6, 6, 7, 4,
            0, 4, 7, 7, 3, 0,
            1, 5, 6, 6, 2, 1,
            0, 1, 5, 5, 4, 0,
            3, 2, 6, 6, 7, 3
        ])

        self.buffer = self.ctx.buffer(data=(vertices + indices).tobytes())
        self.vao = self.ctx.simple_vertex_array(self.prog, self.buffer, 'in_position',
            vertex_buffer=
            index_buffer=self.ctx.buffer(data=bytes(indices)))

        self.rotation = [0.0, 0.0]
        
    def run(self):
        while not self.window.is_closing:
            self.render()
            self.window.swap_buffers()
            
    def render(self):
        self.ctx.viewport = (0, 0, self.width, self.height)
        self.ctx.clear(0.9, 0.9, 0.9)
        
        proj = Matrix44.perspective_projection(90, self.width / self.height, 0.1, 100)
        model = Matrix44.from_translation(Vector3([0.0, 0.0, -3.0])) * Matrix44.from_eulers((self.rotation[0], self.rotation[1], 0))
        
        self.prog['projection'].write(proj.astype('float32').tobytes())
        self.prog['model'].write(model.astype('float32').tobytes())
        
        self.vao.render()

if __name__ == '__main__':
    app = CubeApp()
    app.run()
