"""SceneSpec — declarative description of a physics scene.

This is the IR (intermediate representation) shared by all simulator backends.
It is serializable (JSON / dataclasses), backend-agnostic, and complete enough
to drive PyBullet, Genesis, MuJoCo, or Brax from the same input.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal, Optional


class ShapeType(str, Enum):
    BOX = "box"
    SPHERE = "sphere"
    CYLINDER = "cylinder"
    PLANE = "plane"
    MESH = "mesh"
    SOFT_BODY = "soft"
    FLUID = "fluid"
    CAPSULE = "capsule"


class ForceType(str, Enum):
    GRAVITY = "gravity"
    IMPULSE = "impulse"
    WIND = "wind"
    THRUST = "thrust"


class ConstraintType(str, Enum):
    FIXED = "fixed"
    HINGE = "hinge"
    POINT2POINT = "point2point"
    SLIDER = "slider"


@dataclass
class SceneObject:
    name: str
    shape: ShapeType
    dimensions: tuple[float, ...]
    mass: float = 1.0
    restitution: float = 0.5
    friction: float = 0.5
    initial_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_orientation_quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    is_static: bool = False
    mesh_path: Optional[str] = None
    color_hint: Optional[tuple[float, float, float]] = None
    semantic_label: Optional[str] = None

    def __post_init__(self) -> None:
        if isinstance(self.shape, str):
            self.shape = ShapeType(self.shape)
        if self.is_static:
            self.mass = 0.0

    def needs_genesis(self) -> bool:
        return self.shape in (ShapeType.SOFT_BODY, ShapeType.FLUID)


@dataclass
class SceneForce:
    type: ForceType
    vector: tuple[float, float, float] = (0.0, -9.81, 0.0)
    target: Optional[str] = None
    timing: tuple[float, float] = (0.0, float("inf"))
    magnitude_scale: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.type, str):
            self.type = ForceType(self.type)


@dataclass
class SceneConstraint:
    type: ConstraintType
    object_a: str
    object_b: Optional[str] = None
    anchor_a: tuple[float, float, float] = (0.0, 0.0, 0.0)
    anchor_b: tuple[float, float, float] = (0.0, 0.0, 0.0)
    axis: tuple[float, float, float] = (0.0, 1.0, 0.0)

    def __post_init__(self) -> None:
        if isinstance(self.type, str):
            self.type = ConstraintType(self.type)


@dataclass
class SceneEvent:
    """Scripted timed event: impulse, attachment, detachment."""
    frame: int
    kind: Literal["impulse", "release", "attach", "detach"]
    target: str
    payload: dict = field(default_factory=dict)


@dataclass
class SceneSpec:
    duration_s: float
    fps: int = 24
    objects: list[SceneObject] = field(default_factory=list)
    forces: list[SceneForce] = field(default_factory=list)
    constraints: list[SceneConstraint] = field(default_factory=list)
    events: list[SceneEvent] = field(default_factory=list)
    gravity: tuple[float, float, float] = (0.0, -9.81, 0.0)
    sim_substeps: int = 8
    camera_height_m: float = 1.5
    scene_origin: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def num_frames(self) -> int:
        return int(self.duration_s * self.fps)

    @property
    def dt(self) -> float:
        return 1.0 / self.fps

    def needs_soft_body_backend(self) -> bool:
        return any(o.needs_genesis() for o in self.objects)

    def select_backend_hint(self) -> str:
        if self.needs_soft_body_backend():
            return "genesis"
        if any(o.semantic_label == "humanoid" for o in self.objects):
            return "mujoco"
        return "pybullet"

    def add_gravity_if_missing(self) -> None:
        if not any(f.type == ForceType.GRAVITY for f in self.forces):
            self.forces.append(SceneForce(type=ForceType.GRAVITY, vector=self.gravity))

    def to_json(self) -> str:
        d = asdict(self)
        d["objects"] = [{**asdict(o), "shape": o.shape.value} for o in self.objects]
        d["forces"] = [{**asdict(f), "type": f.type.value} for f in self.forces]
        d["constraints"] = [{**asdict(c), "type": c.type.value} for c in self.constraints]
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "SceneSpec":
        d = json.loads(s)
        d["objects"] = [SceneObject(**o) for o in d.get("objects", [])]
        d["forces"] = [SceneForce(**f) for f in d.get("forces", [])]
        d["constraints"] = [SceneConstraint(**c) for c in d.get("constraints", [])]
        d["events"] = [SceneEvent(**e) for e in d.get("events", [])]
        return cls(**d)

    @classmethod
    def from_file(cls, path: str | Path) -> "SceneSpec":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


def make_falling_object_scene(
    object_name: str = "hero",
    height_m: float = 1.0,
    duration_s: float = 2.0,
    fps: int = 24,
    table_size: tuple[float, float, float] = (1.0, 0.05, 1.0),
    object_size: tuple[float, float, float] = (0.05, 0.1, 0.05),
) -> SceneSpec:
    spec = SceneSpec(
        duration_s=duration_s,
        fps=fps,
        objects=[
            SceneObject(
                name="table",
                shape=ShapeType.BOX,
                dimensions=table_size,
                is_static=True,
                initial_position=(0.0, 0.0, 0.0),
                semantic_label="surface",
            ),
            SceneObject(
                name=object_name,
                shape=ShapeType.CYLINDER,
                dimensions=object_size,
                mass=0.5,
                restitution=0.2,
                friction=0.4,
                initial_position=(0.0, height_m, 0.0),
                semantic_label="hero_product",
            ),
        ],
    )
    spec.add_gravity_if_missing()
    return spec


def make_rolling_object_scene(
    object_name: str = "ball",
    initial_velocity: tuple[float, float, float] = (0.3, 0.0, 0.0),
    duration_s: float = 3.0,
    fps: int = 24,
) -> SceneSpec:
    spec = SceneSpec(
        duration_s=duration_s,
        fps=fps,
        objects=[
            SceneObject(
                name="floor",
                shape=ShapeType.PLANE,
                dimensions=(10.0, 10.0),
                is_static=True,
                friction=0.6,
            ),
            SceneObject(
                name=object_name,
                shape=ShapeType.SPHERE,
                dimensions=(0.08,),
                mass=0.3,
                restitution=0.6,
                friction=0.3,
                initial_position=(0.0, 0.08, 0.0),
                initial_velocity=initial_velocity,
            ),
        ],
    )
    spec.add_gravity_if_missing()
    return spec
