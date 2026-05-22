"""Physics simulator backend interface + concrete PyBullet implementation.

Genesis / MuJoCo / Brax backends provided as stubs with the same contract —
they raise NotImplementedError until their dependencies are installed and
the bridges land in follow-up commits.

All backends produce a `KeyframeTrajectories` from a `SceneSpec`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ogenti.physics.scene import (
    ConstraintType,
    ForceType,
    SceneObject,
    SceneSpec,
    ShapeType,
)
from ogenti.physics.trajectories import KeyframeTrajectories, ObjectTrajectory
from ogenti.utils.logging import get_logger

log = get_logger("ogenti.physics.simulator")


class PhysicsBackend(ABC):
    name: str = "abstract"

    @abstractmethod
    def simulate(self, scene: SceneSpec) -> KeyframeTrajectories: ...


# ────────────────────────────────────────────────────────────────────
# PyBullet backend (default — rigid body)
# ────────────────────────────────────────────────────────────────────


class PyBulletBackend(PhysicsBackend):
    name = "pybullet"

    def __init__(self, gui: bool = False) -> None:
        self.gui = gui
        self._client = None

    def _connect(self) -> int:
        import pybullet as p

        mode = p.GUI if self.gui else p.DIRECT
        client = p.connect(mode)
        p.resetSimulation(physicsClientId=client)
        return client

    def _disconnect(self, client: int) -> None:
        import pybullet as p

        try:
            p.disconnect(physicsClientId=client)
        except Exception:
            pass

    def simulate(self, scene: SceneSpec) -> KeyframeTrajectories:
        try:
            import pybullet as p
        except ImportError as e:
            raise ImportError(
                "pybullet not installed. `pip install pybullet`"
            ) from e

        client = self._connect()
        try:
            return self._simulate_inner(p, client, scene)
        finally:
            self._disconnect(client)

    def _simulate_inner(self, p, client: int, scene: SceneSpec) -> KeyframeTrajectories:
        p.setGravity(*scene.gravity, physicsClientId=client)
        dt = scene.dt / scene.sim_substeps
        p.setTimeStep(dt, physicsClientId=client)

        body_ids: dict[str, int] = {}
        for obj in scene.objects:
            body_ids[obj.name] = self._spawn_object(p, client, obj)

        constraint_handles: list[int] = []
        for c in scene.constraints:
            cid = self._spawn_constraint(p, client, c, body_ids)
            if cid is not None:
                constraint_handles.append(cid)

        traj_pos: dict[str, list[np.ndarray]] = {n: [] for n in body_ids}
        traj_orn: dict[str, list[np.ndarray]] = {n: [] for n in body_ids}
        traj_vel: dict[str, list[np.ndarray]] = {n: [] for n in body_ids}
        traj_ang: dict[str, list[np.ndarray]] = {n: [] for n in body_ids}
        contact_events: dict[str, list[tuple[int, str]]] = {n: [] for n in body_ids}

        ext_forces = [f for f in scene.forces if f.type != ForceType.GRAVITY]

        for frame in range(scene.num_frames):
            self._apply_external_forces(p, client, ext_forces, body_ids, frame, scene)
            self._apply_events(p, client, scene, body_ids, frame)

            for _ in range(scene.sim_substeps):
                p.stepSimulation(physicsClientId=client)

            for name, bid in body_ids.items():
                pos, orn = p.getBasePositionAndOrientation(bid, physicsClientId=client)
                lin, ang = p.getBaseVelocity(bid, physicsClientId=client)
                traj_pos[name].append(np.asarray(pos, dtype=np.float32))
                traj_orn[name].append(np.asarray(orn, dtype=np.float32))
                traj_vel[name].append(np.asarray(lin, dtype=np.float32))
                traj_ang[name].append(np.asarray(ang, dtype=np.float32))

            contacts = p.getContactPoints(physicsClientId=client)
            for c in contacts:
                a_id, b_id = c[1], c[2]
                names = {v: k for k, v in body_ids.items()}
                a_name = names.get(a_id)
                b_name = names.get(b_id)
                if a_name and b_name:
                    contact_events[a_name].append((frame, b_name))

        return self._collect(scene, traj_pos, traj_orn, traj_vel, traj_ang, contact_events)

    def _spawn_object(self, p, client: int, obj: SceneObject) -> int:
        shape_id, visual_id = self._create_shapes(p, client, obj)
        body = p.createMultiBody(
            baseMass=obj.mass,
            baseCollisionShapeIndex=shape_id,
            baseVisualShapeIndex=visual_id,
            basePosition=list(obj.initial_position),
            baseOrientation=list(obj.initial_orientation_quat),
            physicsClientId=client,
        )
        p.changeDynamics(
            body, -1,
            restitution=obj.restitution,
            lateralFriction=obj.friction,
            physicsClientId=client,
        )
        if not obj.is_static:
            p.resetBaseVelocity(
                body, list(obj.initial_velocity), list(obj.initial_angular_velocity),
                physicsClientId=client,
            )
        return body

    def _create_shapes(self, p, client: int, obj: SceneObject) -> tuple[int, int]:
        if obj.shape == ShapeType.BOX:
            half = [d / 2.0 for d in obj.dimensions[:3]]
            shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=client)
            visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half, physicsClientId=client)
        elif obj.shape == ShapeType.SPHERE:
            r = obj.dimensions[0]
            shape = p.createCollisionShape(p.GEOM_SPHERE, radius=r, physicsClientId=client)
            visual = p.createVisualShape(p.GEOM_SPHERE, radius=r, physicsClientId=client)
        elif obj.shape == ShapeType.CYLINDER:
            r = obj.dimensions[0]
            h = obj.dimensions[1] if len(obj.dimensions) > 1 else 2 * r
            shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=h, physicsClientId=client)
            visual = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=h, physicsClientId=client)
        elif obj.shape == ShapeType.CAPSULE:
            r = obj.dimensions[0]
            h = obj.dimensions[1] if len(obj.dimensions) > 1 else 2 * r
            shape = p.createCollisionShape(p.GEOM_CAPSULE, radius=r, height=h, physicsClientId=client)
            visual = p.createVisualShape(p.GEOM_CAPSULE, radius=r, length=h, physicsClientId=client)
        elif obj.shape == ShapeType.PLANE:
            shape = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=client)
            visual = p.createVisualShape(p.GEOM_PLANE, physicsClientId=client)
        elif obj.shape == ShapeType.MESH and obj.mesh_path:
            shape = p.createCollisionShape(p.GEOM_MESH, fileName=obj.mesh_path, physicsClientId=client)
            visual = p.createVisualShape(p.GEOM_MESH, fileName=obj.mesh_path, physicsClientId=client)
        else:
            log.warning(f"unsupported shape {obj.shape} in PyBullet, falling back to box")
            half = [0.05, 0.05, 0.05]
            shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=client)
            visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half, physicsClientId=client)
        return shape, visual

    def _spawn_constraint(self, p, client: int, c, body_ids: dict[str, int]) -> Optional[int]:
        if c.object_a not in body_ids:
            return None
        bid_a = body_ids[c.object_a]
        bid_b = body_ids.get(c.object_b, -1) if c.object_b else -1

        if c.type == ConstraintType.FIXED:
            jt = p.JOINT_FIXED
        elif c.type == ConstraintType.HINGE:
            jt = p.JOINT_REVOLUTE
        elif c.type == ConstraintType.SLIDER:
            jt = p.JOINT_PRISMATIC
        else:
            jt = p.JOINT_POINT2POINT

        return p.createConstraint(
            parentBodyUniqueId=bid_a,
            parentLinkIndex=-1,
            childBodyUniqueId=bid_b,
            childLinkIndex=-1,
            jointType=jt,
            jointAxis=list(c.axis),
            parentFramePosition=list(c.anchor_a),
            childFramePosition=list(c.anchor_b),
            physicsClientId=client,
        )

    def _apply_external_forces(
        self, p, client: int, forces, body_ids: dict[str, int], frame: int, scene: SceneSpec
    ) -> None:
        t_now = frame * scene.dt
        for f in forces:
            t0, t1 = f.timing
            if not (t0 <= t_now < t1):
                continue
            scaled = tuple(v * f.magnitude_scale for v in f.vector)
            targets = [f.target] if f.target else list(body_ids.keys())
            for name in targets:
                bid = body_ids.get(name)
                if bid is None:
                    continue
                if f.type == ForceType.IMPULSE and frame == int(t0 / scene.dt):
                    p.applyExternalForce(bid, -1, scaled, [0, 0, 0], p.LINK_FRAME, physicsClientId=client)
                elif f.type in (ForceType.WIND, ForceType.THRUST):
                    p.applyExternalForce(bid, -1, scaled, [0, 0, 0], p.WORLD_FRAME, physicsClientId=client)

    def _apply_events(self, p, client: int, scene: SceneSpec, body_ids: dict[str, int], frame: int) -> None:
        for ev in scene.events:
            if ev.frame != frame:
                continue
            bid = body_ids.get(ev.target)
            if bid is None:
                continue
            if ev.kind == "impulse":
                impulse = ev.payload.get("impulse", [0, 0, 0])
                p.applyExternalForce(bid, -1, impulse, [0, 0, 0], p.LINK_FRAME, physicsClientId=client)

    def _collect(self, scene, pos, orn, vel, ang, contacts) -> KeyframeTrajectories:
        out: list[ObjectTrajectory] = []
        for obj in scene.objects:
            name = obj.name
            p_arr = np.stack(pos[name]) if pos[name] else np.zeros((scene.num_frames, 3), dtype=np.float32)
            o_arr = np.stack(orn[name]) if orn[name] else np.tile(
                np.array([0, 0, 0, 1], dtype=np.float32), (scene.num_frames, 1)
            )
            v_arr = np.stack(vel[name]) if vel[name] else np.zeros((scene.num_frames, 3), dtype=np.float32)
            a_arr = np.stack(ang[name]) if ang[name] else np.zeros((scene.num_frames, 3), dtype=np.float32)
            out.append(ObjectTrajectory(
                name=name,
                positions=p_arr,
                orientations=o_arr,
                velocities=v_arr,
                angular_velocities=a_arr,
                visibility=np.ones(scene.num_frames, dtype=np.float32),
                contact_events=contacts.get(name, []),
                semantic_label=obj.semantic_label,
            ))
        return KeyframeTrajectories(
            fps=scene.fps,
            duration_s=scene.duration_s,
            objects=out,
            backend=self.name,
        )


# ────────────────────────────────────────────────────────────────────
# Genesis backend (stub — GPU, soft body + fluid)
# ────────────────────────────────────────────────────────────────────


class GenesisBackend(PhysicsBackend):
    name = "genesis"

    def simulate(self, scene: SceneSpec) -> KeyframeTrajectories:
        try:
            import genesis  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "genesis-world not installed. Use PyBulletBackend or "
                "install genesis: `pip install genesis-world`"
            ) from e
        raise NotImplementedError(
            "GenesisBackend.simulate scaffold — wire to genesis Scene/Sim in v2"
        )


# ────────────────────────────────────────────────────────────────────
# MuJoCo backend (stub — articulated)
# ────────────────────────────────────────────────────────────────────


class MuJoCoBackend(PhysicsBackend):
    name = "mujoco"

    def simulate(self, scene: SceneSpec) -> KeyframeTrajectories:
        try:
            import mujoco  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "mujoco not installed. `pip install mujoco`"
            ) from e
        raise NotImplementedError("MuJoCoBackend.simulate scaffold — wire in v2")


# ────────────────────────────────────────────────────────────────────
# Brax backend (stub — JAX differentiable)
# ────────────────────────────────────────────────────────────────────


class BraxBackend(PhysicsBackend):
    name = "brax"

    def simulate(self, scene: SceneSpec) -> KeyframeTrajectories:
        try:
            import brax  # noqa: F401
        except ImportError as e:
            raise ImportError("brax not installed. `pip install brax`") from e
        raise NotImplementedError("BraxBackend.simulate scaffold — wire in v2")


# ────────────────────────────────────────────────────────────────────
# Auto-select
# ────────────────────────────────────────────────────────────────────


_BACKEND_REGISTRY: dict[str, type[PhysicsBackend]] = {
    "pybullet": PyBulletBackend,
    "genesis": GenesisBackend,
    "mujoco": MuJoCoBackend,
    "brax": BraxBackend,
}


def get_backend(name: str = "auto", scene: Optional[SceneSpec] = None) -> PhysicsBackend:
    if name == "auto":
        if scene is None:
            name = "pybullet"
        else:
            name = scene.select_backend_hint()
    cls = _BACKEND_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"unknown backend '{name}'. known: {list(_BACKEND_REGISTRY)}")
    log.info(f"using physics backend: {name}")
    return cls()


def simulate(scene: SceneSpec, backend: str = "auto") -> KeyframeTrajectories:
    return get_backend(backend, scene).simulate(scene)
