import numpy as np


class CameraProjection:
    _IMAGE_CORNER_LABELS = ("tl", "tr", "br", "bl")
    _CAMERA_TO_WORLD_0 = np.array(
        [
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0],
        ],
        dtype=float,
    )

    def __init__(
        self,
        camera_matrix,
        utm_location,
        agl,
        heading,
        pitch,
        yaw=0.0,
        camera_matrix_inv=None,
        center_ray=None,
        corner_rays=None,
    ):
        self.K = camera_matrix if isinstance(camera_matrix, np.ndarray) else np.array(camera_matrix, dtype=float)
        self.utm = utm_location
        self.agl = float(agl)
        self.heading = float(heading)
        self.pitch = float(pitch)
        self.yaw = float(yaw)
        if camera_matrix_inv is None:
            self._K_inv = np.linalg.inv(self.K)
        else:
            self._K_inv = (
                camera_matrix_inv
                if isinstance(camera_matrix_inv, np.ndarray)
                else np.array(camera_matrix_inv, dtype=float)
            )
        self._center_ray = center_ray
        self._corner_rays = corner_rays
        if self._center_ray is None or self._corner_rays is None:
            self._center_ray = self._pixel_to_camera_ray(self.K[0, 2], self.K[1, 2])
            self._corner_rays = tuple(
                self._pixel_to_camera_ray(u, v)
                for u, v in self._image_corners()
            )

    def project(self):
        corners = {}
        cam_to_world = self._rotation_matrix(self.heading, self.pitch, self.yaw) @ self._CAMERA_TO_WORLD_0
        for label, ray in zip(self._IMAGE_CORNER_LABELS, self._corner_rays):
            corners[label] = self._ray_to_ground(ray, cam_to_world)
        return corners

    def project_center(self):
        cam_to_world = self._rotation_matrix(self.heading, self.pitch, self.yaw) @ self._CAMERA_TO_WORLD_0
        return self._ray_to_ground(self._center_ray, cam_to_world)

    def center_ray_world(self):
        cam_to_world = self._rotation_matrix(self.heading, self.pitch, self.yaw) @ self._CAMERA_TO_WORLD_0
        return cam_to_world @ self._center_ray

    def _image_corners(self):
        cx, cy = self.K[0, 2], self.K[1, 2]
        width, height = 2 * cx, 2 * cy
        return [
            (0, 0),
            (width - 1, 0),
            (width - 1, height - 1),
            (0, height - 1),
        ]

    def _pixel_to_camera_ray(self, u, v):
        point = self._K_inv @ np.array([u, v, 1.0])
        return point / np.linalg.norm(point)

    @classmethod
    def precompute_rays(cls, camera_matrix, camera_matrix_inv=None):
        camera_matrix = (
            camera_matrix
            if isinstance(camera_matrix, np.ndarray)
            else np.array(camera_matrix, dtype=float)
        )
        if camera_matrix_inv is None:
            camera_matrix_inv = np.linalg.inv(camera_matrix)
        elif not isinstance(camera_matrix_inv, np.ndarray):
            camera_matrix_inv = np.array(camera_matrix_inv, dtype=float)

        helper = cls(
            camera_matrix,
            (0.0, 0.0),
            1.0,
            0.0,
            -90.0,
            camera_matrix_inv=camera_matrix_inv,
        )
        return helper._center_ray, helper._corner_rays

    @staticmethod
    def _rotation_matrix(heading_deg, pitch_deg, yaw_deg=0.0):
        heading = np.radians(heading_deg + yaw_deg)
        pitch = np.radians(pitch_deg)

        rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(pitch), -np.sin(pitch)],
                [0, np.sin(pitch), np.cos(pitch)],
            ]
        )
        rz = np.array(
            [
                [np.cos(heading), np.sin(heading), 0],
                [-np.sin(heading), np.cos(heading), 0],
                [0, 0, 1],
            ]
        )
        return rz @ rx

    def _ray_to_ground(self, ray_camera, cam_to_world=None):
        if cam_to_world is None:
            cam_to_world = self._rotation_matrix(self.heading, self.pitch, self.yaw) @ self._CAMERA_TO_WORLD_0

        ray_world = cam_to_world @ ray_camera
        rz = ray_world[2]
        if abs(rz) < 1e-9:
            raise ValueError("Ray is nearly parallel to the ground.")
        if rz > 0:
            raise ValueError("Ray points upward; check pitch and heading.")

        t_value = -self.agl / rz
        easting = self.utm[0] + t_value * ray_world[0]
        northing = self.utm[1] + t_value * ray_world[1]
        return (easting, northing)
