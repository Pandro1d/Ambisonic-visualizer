import numpy as np
import cv2
import matplotlib.pyplot as plt 
from matplotlib import colormaps
from scipy.spatial import cKDTree

from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import *
from OpenGL.GLU import gluPerspective, gluLookAt

def create_sphere_mesh_fast(radius, slices, stacks):
    vertices = []
    uvs = []

    for i in range(stacks + 1):
        lat = np.pi * (-0.5 + float(i) / stacks)
        y0 = np.sin(lat)
        zr = np.cos(lat)
        if i == 0 or i == stacks: zr = 0.0

        for j in range(slices + 1):
            lng = 2 * np.pi * float(j) / slices
            x0 = zr * np.cos(lng)
            z0 = zr * np.sin(lng)
            vertices.append([x0 * radius, y0 * radius, z0 * radius])
            
            u = float(j) / slices # Bez "1.0 - " (naprawa lustrzanego odbicia)
            v = 1.0 - (float(i) / stacks) 
            uvs.append([u, v])

    vertices = np.array(vertices, dtype=np.float32)
    uvs = np.array(uvs, dtype=np.float32)

    indices = []
    stride = slices + 1
    for i in range(stacks):
        for j in range(slices):
            p1 = i * stride + j
            p2 = p1 + 1
            p3 = (i + 1) * stride + j + 1
            p4 = (i + 1) * stride + j
            indices.extend([p1, p2, p3, p4])

    indices = np.array(indices, dtype=np.uint32)
    return vertices, uvs, indices

class VideoHandler:
    def __init__(self):
        self.cap = None
        self.image = None
        self.is_video = False
        self.fps = 0
        self.frame_count = 0

    def load(self, path):
        self.cap = cv2.VideoCapture(path)
        if self.cap.isOpened():
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.is_video = True
            return True
        img = cv2.imread(path)
        if img is not None:
            self.is_video = False
            self.image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return True
        return False

    def get_frame(self, t_sec):
        if not self.is_video: return self.image
        target = int(t_sec * self.fps)
        if target >= self.frame_count: target = target % self.frame_count
        curr = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        if abs(curr - target) > 5: self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = self.cap.read()
        if ret: return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret else None

class FPSGLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_obj = parent 
        self.max_energy = 0.5 # Domyślna czułość (auto-kalibracja mile widziana)
        self.sharpness = 1.0
        self.slices = 80 
        self.stacks = 60
        self.verts = None
        self.uvs = None
        self.indices = None
        self.vertex_indices = None
        self.colors = None
        self.yaw = -90.0
        self.pitch = 0.0
        self.lastPos = QPoint()
        self.sensitivity = 0.2
        self.fov = 100.0
        self.texture_id = None
        try: self.cmap = colormaps['jet']
        except: self.cmap = plt.get_cmap('jet')

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
        self.verts, self.uvs, self.indices = create_sphere_mesh_fast(50.0, self.slices, self.stacks)

    def prepare_mesh_mapping(self, audio_xyz_coords):
        if self.verts is None: return
        norms = np.linalg.norm(self.verts, axis=1, keepdims=True)
        norms[norms == 0] = 1
        unit_mesh = self.verts / norms
        tree = cKDTree(audio_xyz_coords)
        _, self.vertex_indices = tree.query(unit_mesh)

    def update_texture(self, img_data):
        if img_data is None: return
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        h, w, c = img_data.shape
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)

    def set_energy_data(self, energy_snapshot):
        """Odbiera obliczoną energię z wątku audio w czasie rzeczywistym"""
        if self.vertex_indices is None: return
        
        # Mapowanie punktów obliczeń na siatkę sfery
        mesh_energies = energy_snapshot[self.vertex_indices]
        
        # Prosta auto-kalibracja max_energy (zanikanie powolne)
        curr_max = np.max(mesh_energies)
        if curr_max > self.max_energy:
            self.max_energy = curr_max
        else:
            self.max_energy *= 0.99 # Powolny powrót
            
        norm_e = np.clip(mesh_energies / (self.max_energy + 1e-9), 0.0, 1.0)
        sharpened_e = np.power(norm_e, self.sharpness)
        rgba = self.cmap(sharpened_e)
        rgba[:, 3] = rgba[:, 3] * sharpened_e * 0.9 
        self.colors = np.ascontiguousarray(rgba, dtype=np.float32)
        self.update()

    def set_sharpness(self, val):
        self.sharpness = val

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.fov, w / max(1.0, h), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        rad_yaw = np.radians(self.yaw)
        rad_pitch = np.radians(self.pitch)
        look_x = np.cos(rad_yaw) * np.cos(rad_pitch)
        look_y = np.sin(rad_pitch)
        look_z = np.sin(rad_yaw) * np.cos(rad_pitch)
        gluLookAt(0, 0, 0, look_x, look_y, look_z, 0, 1, 0)
        
        if self.verts is None: return
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_TEXTURE_COORD_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, self.verts)
        glTexCoordPointer(2, GL_FLOAT, 0, self.uvs)

        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glColor3f(1.0, 1.0, 1.0)
        glDrawElements(GL_QUADS, len(self.indices), GL_UNSIGNED_INT, self.indices)
        glDisable(GL_TEXTURE_2D)

        if self.colors is not None:
            glDisable(GL_DEPTH_TEST)
            glEnable(GL_BLEND)
            glEnableClientState(GL_COLOR_ARRAY)
            glPushMatrix()
            glScalef(0.99, 0.99, 0.99) 
            glColorPointer(4, GL_FLOAT, 0, self.colors)
            glDrawElements(GL_QUADS, len(self.indices), GL_UNSIGNED_INT, self.indices)
            glPopMatrix()
            glDisableClientState(GL_COLOR_ARRAY)
            glDisable(GL_BLEND)
            glEnable(GL_DEPTH_TEST)
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_TEXTURE_COORD_ARRAY)

    def mousePressEvent(self, event): self.lastPos = event.pos()
    def mouseMoveEvent(self, event):
        dx = event.pos().x() - self.lastPos.x()
        dy = event.pos().y() - self.lastPos.y()
        self.lastPos = event.pos()
        self.yaw += dx * self.sensitivity
        self.pitch -= dy * self.sensitivity
        self.pitch = max(-89.0, min(89.0, self.pitch))
        self.update()
        if self.parent_obj and hasattr(self.parent_obj, 'update_audio_rotation'):
            self.parent_obj.update_audio_rotation(self.yaw, self.pitch)