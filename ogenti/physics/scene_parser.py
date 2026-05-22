"""Prompt → SceneSpec.

Two-tier parser:
  1. Heuristic template matching against an ad-creative pattern library.
  2. LLM fallback (Anthropic / OpenAI) for prompts that don't match a template.

If both fail we emit a minimal scene (ground plane + hero on table + gravity).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Callable, Optional

from ogenti.physics.scene import (
    SceneObject,
    SceneSpec,
    ShapeType,
    make_falling_object_scene,
    make_rolling_object_scene,
)
from ogenti.utils.logging import get_logger

log = get_logger("ogenti.physics.scene_parser")


@dataclass
class TemplateMatch:
    name: str
    pattern: re.Pattern
    builder: Callable[[re.Match, str], SceneSpec]
    priority: int = 0


# ────────────────────────────────────────────────────────────────────
# Heuristic templates
# ────────────────────────────────────────────────────────────────────


def _template_falling(match: re.Match, prompt: str) -> SceneSpec:
    object_name = match.group("obj")[:32] if match.group("obj") else "hero"
    return make_falling_object_scene(object_name=object_name)


def _template_rolling(match: re.Match, prompt: str) -> SceneSpec:
    object_name = match.group("obj")[:32] if match.group("obj") else "ball"
    direction = (0.3, 0.0, 0.0)
    if "right" in prompt.lower():
        direction = (0.3, 0.0, 0.0)
    elif "left" in prompt.lower():
        direction = (-0.3, 0.0, 0.0)
    return make_rolling_object_scene(object_name=object_name, initial_velocity=direction)


def _template_splash(match: re.Match, prompt: str) -> SceneSpec:
    spec = SceneSpec(
        duration_s=2.0,
        fps=24,
        objects=[
            SceneObject(name="cup", shape=ShapeType.CYLINDER, dimensions=(0.05, 0.1),
                        is_static=True, initial_position=(0, 0.05, 0), semantic_label="container"),
            SceneObject(name="liquid", shape=ShapeType.FLUID, dimensions=(0.04, 0.06, 0.04),
                        mass=0.1, initial_position=(0, 0.08, 0), semantic_label="liquid"),
            SceneObject(name="drop", shape=ShapeType.SPHERE, dimensions=(0.01,),
                        mass=0.005, initial_position=(0, 0.4, 0),
                        initial_velocity=(0, -1.5, 0), semantic_label="droplet"),
        ],
    )
    spec.add_gravity_if_missing()
    return spec


def _template_cloth_drape(match: re.Match, prompt: str) -> SceneSpec:
    spec = SceneSpec(
        duration_s=2.5,
        fps=24,
        objects=[
            SceneObject(name="furniture", shape=ShapeType.BOX, dimensions=(0.4, 0.3, 0.4),
                        is_static=True, initial_position=(0, 0.15, 0)),
            SceneObject(name="cloth", shape=ShapeType.SOFT_BODY, dimensions=(0.6, 0.6),
                        mass=0.3, initial_position=(0, 0.6, 0), semantic_label="fabric"),
        ],
    )
    spec.add_gravity_if_missing()
    return spec


def _template_swing(match: re.Match, prompt: str) -> SceneSpec:
    from ogenti.physics.scene import SceneConstraint, ConstraintType

    spec = SceneSpec(
        duration_s=3.0,
        fps=24,
        objects=[
            SceneObject(name="anchor", shape=ShapeType.SPHERE, dimensions=(0.02,),
                        is_static=True, initial_position=(0, 1.0, 0)),
            SceneObject(name="bob", shape=ShapeType.SPHERE, dimensions=(0.05,),
                        mass=0.5, initial_position=(0.4, 1.0, 0),
                        initial_velocity=(0, 0, 0)),
        ],
        constraints=[
            SceneConstraint(type=ConstraintType.POINT2POINT, object_a="anchor", object_b="bob",
                            anchor_b=(-0.4, 0, 0)),
        ],
    )
    spec.add_gravity_if_missing()
    return spec


TEMPLATES: list[TemplateMatch] = [
    TemplateMatch(
        name="splash",
        pattern=re.compile(r"\b(splash|pour|spill|drip|droplet)\b", re.IGNORECASE),
        builder=_template_splash,
        priority=10,
    ),
    TemplateMatch(
        name="cloth_drape",
        pattern=re.compile(r"\b(drape|cloth|fabric|cover|veil)\b", re.IGNORECASE),
        builder=_template_cloth_drape,
        priority=8,
    ),
    TemplateMatch(
        name="swing",
        pattern=re.compile(r"\b(swing|pendulum|hang|dangle)\b", re.IGNORECASE),
        builder=_template_swing,
        priority=6,
    ),
    TemplateMatch(
        name="falling",
        pattern=re.compile(r"(?P<obj>\w+(?:\s+\w+){0,2}?)\s+(?:falls|drops|tumbles|tips)", re.IGNORECASE),
        builder=_template_falling,
        priority=5,
    ),
    TemplateMatch(
        name="rolling",
        pattern=re.compile(r"(?P<obj>\w+(?:\s+\w+){0,2}?)\s+(?:rolls|skids|slides)", re.IGNORECASE),
        builder=_template_rolling,
        priority=5,
    ),
]


def parse_prompt_heuristic(prompt: str) -> Optional[SceneSpec]:
    candidates: list[tuple[TemplateMatch, re.Match]] = []
    for tpl in TEMPLATES:
        m = tpl.pattern.search(prompt)
        if m:
            candidates.append((tpl, m))
    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[0].priority)
    tpl, m = candidates[0]
    log.info(f"matched template '{tpl.name}' for prompt: {prompt[:60]}...")
    return tpl.builder(m, prompt)


# ────────────────────────────────────────────────────────────────────
# LLM fallback
# ────────────────────────────────────────────────────────────────────


_LLM_SYSTEM_PROMPT = """You are a physics scene specification generator.
Given a natural-language prompt describing a video shot, output a JSON
SceneSpec compatible with the ogenti.physics.scene module.

Required fields:
  duration_s: float (seconds)
  fps: int (default 24)
  objects: array of SceneObject:
    {
      "name": "string",
      "shape": "box|sphere|cylinder|plane|mesh|soft|fluid|capsule",
      "dimensions": [float, ...],   # box: [w,h,d], sphere: [r], cylinder: [r,h]
      "mass": float,                # 0 if static
      "restitution": 0.0..1.0,
      "friction": 0.0..1.0,
      "initial_position": [x,y,z],
      "initial_velocity": [x,y,z],
      "initial_orientation_quat": [x,y,z,w],
      "is_static": bool,
      "semantic_label": "string"
    }
  forces: array of SceneForce:
    {
      "type": "gravity|impulse|wind|thrust",
      "vector": [x,y,z],
      "target": "object_name" or null
    }

Use SI units (meters, kg, seconds). +y is up. The camera sits at y=1.5 m.

Respond ONLY with valid JSON. No prose."""


def parse_prompt_llm(
    prompt: str,
    model: str = "claude-3-5-sonnet-latest",
    timeout_s: float = 20.0,
) -> Optional[SceneSpec]:
    """LLM-driven SceneSpec generation. Cached on disk."""
    cache_dir = os.environ.get("OGENTI_SCENE_CACHE", ".ogenti_scene_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = _cache_key(prompt, model)
    cache_path = os.path.join(cache_dir, f"{cache_key}.json")

    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return SceneSpec.from_json(f.read())
        except Exception:
            pass

    raw = _call_llm(prompt, model, timeout_s)
    if raw is None:
        return None

    try:
        spec = SceneSpec.from_json(raw)
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(raw)
        return spec
    except Exception as e:
        log.warning(f"LLM returned invalid SceneSpec JSON: {e}")
        return None


def _call_llm(prompt: str, model: str, timeout_s: float) -> Optional[str]:
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            client = anthropic.Anthropic()
            msg = client.messages.create(
                model=model,
                max_tokens=2048,
                system=_LLM_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return _extract_json(msg.content[0].text)
        except Exception as e:
            log.warning(f"Anthropic LLM call failed: {e}")

    if os.environ.get("OPENAI_API_KEY"):
        try:
            import openai
            client = openai.OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                timeout=timeout_s,
            )
            return resp.choices[0].message.content
        except Exception as e:
            log.warning(f"OpenAI LLM call failed: {e}")

    return None


def _extract_json(text: str) -> str:
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return fence.group(1)
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        return text[brace_start : brace_end + 1]
    return text


def _cache_key(prompt: str, model: str) -> str:
    import hashlib
    h = hashlib.sha256(f"{model}\n{prompt}".encode("utf-8")).hexdigest()
    return h[:32]


# ────────────────────────────────────────────────────────────────────
# Top-level
# ────────────────────────────────────────────────────────────────────


def parse_prompt(
    prompt: str,
    use_llm_fallback: bool = True,
    default_factory: Callable[[], SceneSpec] = make_falling_object_scene,
) -> SceneSpec:
    spec = parse_prompt_heuristic(prompt)
    if spec is not None:
        return spec

    if use_llm_fallback:
        spec = parse_prompt_llm(prompt)
        if spec is not None:
            return spec

    log.info("no template match, no LLM available — using default scene")
    return default_factory()
