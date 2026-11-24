"""Tests for the approach skill against a fake world.

The fake keeps the robot pose and one object. Turns overshoot by 40 % (the real
base turned 20-60 % too far), drives are exact, and each camera sees the object
only inside its field of view and range.
"""

import asyncio
import json
import math

import pytest

from skill_based.skills import approach
from test_pick import DONE

HALF_DEPTH = 0.04  # object centroid behind its near face
FOV = {"front": (math.radians(30), 0.4, 6.0), "arm": (math.radians(25), 0.8, 3.0)}


class FakeWorld:
    def __init__(self, obj=(2.0, 0.0), name="green cube", overshoot=1.4):
        self.x = self.y = self.yaw = 0.0
        self.obj = obj
        self.name = name
        self.overshoot = overshoot
        self.cached = None
        self.calls = []
        self.blind_below = 0.0  # neither camera sees the object closer than this
        self.ghost = None  # after the first drive SAM3 picks this base-frame (x, y) instead
        self.drives = 0

    def _in_base(self):
        dx, dy = self.obj[0] - self.x, self.obj[1] - self.y
        c, s = math.cos(-self.yaw), math.sin(-self.yaw)
        return c * dx - s * dy, s * dx + c * dy

    def near_face(self):
        bx, by = self._in_base()
        return math.hypot(bx, by) - HALF_DEPTH

    def bearing(self):
        bx, by = self._in_base()
        return math.atan2(by, bx)

    async def call_tool_prefixed(self, name, args):
        self.calls.append((name, args))
        if name == "moveit__plan_and_execute":
            return DONE
        if name == "perception__segment_objects":
            bx, by = self._in_base()
            half_fov, near, far = FOV[args["camera"]]
            seen = (args["prompt"] == self.name and abs(math.atan2(by, bx)) <= half_fov
                    and max(near, self.blind_below) <= math.hypot(bx, by) <= far)
            if self.ghost and self.drives:
                bx, by = self.ghost
                seen = True
            self.cached = (bx, by) if seen else None
            return json.dumps({"status": "SUCCESS" if seen else "NO_OBJECTS_FOUND"})
        if name == "perception__get_topdown_grasp_pose":
            bx, by = self.cached
            return json.dumps({
                "centroid_base_frame": {"x": bx, "y": by, "z": 0.04},
                "bbox_base_frame": {"x_min": bx - HALF_DEPTH, "x_max": bx + HALF_DEPTH},
            })
        if name == "nav2__spin_robot":
            self.yaw += args["angle"] * self.overshoot
            return f"Successfully spun robot by {args['angle']:.2f} radians"
        if name in ("nav2__drive_on_heading", "nav2__backup_robot"):
            d = args["distance"] if name == "nav2__drive_on_heading" else -args["distance"]
            self.drives += 1
            self.x += d * math.cos(self.yaw)
            self.y += d * math.sin(self.yaw)
            return f"Successfully drove {args['distance']:.2f} meters on heading"
        return "ok"

    def names(self):
        return [n for n, _ in self.calls]


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    async def still(_mcp):
        return None

    monkeypatch.setattr(approach, "wait_until_still", still)


def _run(world, next_action="pick"):
    return asyncio.run(approach.run(world, world.name, next_action))


def _at_standoff(world, next_action="pick"):
    return approach._in_window(world.near_face(), next_action)


def test_object_straight_ahead_ends_at_the_standoff():
    world = FakeWorld(obj=(2.0, 0.0))
    result = _run(world)
    assert result["success"], result["reason"]
    assert _at_standoff(world)
    assert "nav2__spin_robot" not in world.names()


def test_object_to_the_side_is_faced_despite_overshooting_turns():
    world = FakeWorld(obj=(1.6, 1.3))  # about 39 deg left
    result = _run(world, "container_place")
    assert result["success"], result["reason"]
    assert _at_standoff(world, "container_place")
    assert abs(world.bearing()) <= approach.BEARING_TOL_RAD
    assert world.names().count("nav2__spin_robot") >= 2


def test_base_that_turns_twice_as_far_still_faces_the_object():
    world = FakeWorld(obj=(1.7, -0.45), overshoot=2.0)  # about 15 deg right, as on the robot
    result = _run(world, "container_place")
    assert result["success"], result["reason"]
    assert abs(world.bearing()) <= approach.BEARING_TOL_RAD
    assert _at_standoff(world, "container_place")


def test_object_behind_is_found_by_turning():
    world = FakeWorld(obj=(-2.0, 0.3))
    result = _run(world)
    assert result["success"], result["reason"]
    assert _at_standoff(world)


def test_missing_object_fails_without_driving():
    world = FakeWorld(name="green cube")
    result = asyncio.run(approach.run(world, "red ball", "pick"))
    assert not result["success"]
    assert "not found" in result["reason"]
    assert "nav2__drive_on_heading" not in world.names()
    assert world.names().count("nav2__backup_robot") == 1  # the one look from farther back


def test_object_beyond_range_is_not_the_target():
    world = FakeWorld(obj=(5.0, 0.0))
    result = _run(world)
    assert not result["success"]
    assert "nav2__drive_on_heading" not in world.names()


def test_already_at_the_standoff_does_not_move_the_base():
    world = FakeWorld(obj=(0.85 + HALF_DEPTH, 0.0))
    result = _run(world)
    assert result["success"], result["reason"]
    assert not any(n.startswith("nav2__") for n in world.names())


def test_too_close_backs_up():
    world = FakeWorld(obj=(0.5 + HALF_DEPTH, 0.0))  # near face 0.5 m, floor_place wants 0.85
    result = _run(world, "floor_place")
    assert result["success"], result["reason"]
    assert "nav2__backup_robot" in world.names()
    assert _at_standoff(world, "floor_place")


def test_unknown_next_action_does_nothing():
    world = FakeWorld()
    result = _run(world, "throw")
    assert not result["success"]
    assert world.calls == []


def test_failed_turn_stops_the_skill():
    world = FakeWorld(obj=(1.6, 1.3))

    async def refuse(name, args, _orig=world.call_tool_prefixed):
        if name == "nav2__spin_robot":
            world.calls.append((name, args))
            return "Error: Spin operation failed"
        return await _orig(name, args)

    world.call_tool_prefixed = refuse
    result = _run(world)
    assert not result["success"]
    assert "turn failed" in result["reason"]
    assert "nav2__drive_on_heading" not in world.names()


def test_window_far_ends_follow_the_reach_of_pick_and_place():
    assert approach.MAX_NEAR_BY_NEXT_ACTION == pytest.approx(
        {"pick": 1.00, "container_place": 0.80, "surface_place": 0.85, "floor_place": 1.00})


def test_object_too_close_for_the_cameras_counts_as_reached_by_odometry():
    world = FakeWorld(obj=(2.0, 0.0))
    world.blind_below = 1.0  # low container: lost from view at the standoff
    result = _run(world, "container_place")
    assert result["success"], result["reason"]
    assert result["verified"] is False
    assert _at_standoff(world, "container_place")


def test_lost_close_to_the_standoff_drives_the_rest_blind():
    world = FakeWorld(obj=(1.9 + HALF_DEPTH, 0.0))  # first leg 0.8 m leaves 0.4 m to go
    world.blind_below = 1.2
    result = _run(world, "surface_place")
    assert result["success"], result["reason"]
    assert result["verified"] is False
    assert world.names().count("nav2__drive_on_heading") == 2
    assert _at_standoff(world, "surface_place")


def test_object_lost_far_from_the_standoff_still_fails():
    world = FakeWorld(obj=(3.5, 0.0))  # first leg 0.8 m ends ~1.8 m short of the standoff
    world.blind_below = 3.0
    result = _run(world)
    assert not result["success"]
    assert "lost" in result["reason"]
    assert world.names().count("nav2__drive_on_heading") == 1


def test_far_wrong_hit_after_the_drive_is_ignored():
    world = FakeWorld(obj=(2.0, 0.0))
    world.ghost = (4.9, 0.0)  # SAM3 picks another grey object far behind
    result = _run(world, "surface_place")
    assert result["success"], result["reason"]
    assert result["verified"] is False
    assert _at_standoff(world, "surface_place")  # not driven on towards the wrong hit


def test_a_bit_farther_than_the_standoff_is_close_enough():
    world = FakeWorld(obj=(0.77 + HALF_DEPTH, 0.0))  # approach_7: 0.77 m, surface wants 0.70
    result = _run(world, "surface_place")
    assert result["success"], result["reason"]
    assert result["verified"] is True
    assert not any(n.startswith("nav2__") for n in world.names())


def test_pick_reaches_from_almost_a_metre():
    world = FakeWorld(obj=(0.98 + HALF_DEPTH, 0.0))
    result = _run(world, "pick")
    assert result["success"], result["reason"]
    assert not any(n.startswith("nav2__") for n in world.names())


def test_beyond_the_window_drives_one_more_leg():
    world = FakeWorld(obj=(0.90 + HALF_DEPTH, 0.0))  # surface place reaches only from 0.85 m
    result = _run(world, "surface_place")
    assert result["success"], result["reason"]
    assert world.names().count("nav2__drive_on_heading") == 1
    assert _at_standoff(world, "surface_place")


def test_long_way_is_driven_in_short_legs():
    world = FakeWorld(obj=(3.0, 0.0))
    result = _run(world)
    assert result["success"], result["reason"]
    legs = [a["distance"] for n, a in world.calls if n == "nav2__drive_on_heading"]
    assert len(legs) >= 3 and max(legs) <= approach.MAX_LEG_M


def test_too_close_to_see_backs_up_before_searching():
    world = FakeWorld(obj=(0.45 + HALF_DEPTH, 0.0))  # approach_15: grey box at 0.45 m, out of both cameras
    world.blind_below = 0.6
    result = _run(world, "surface_place")
    assert result["success"], result["reason"]
    assert result["verified"] is True
    assert "nav2__spin_robot" not in world.names()
    assert _at_standoff(world, "surface_place")


def test_refused_backup_still_searches():
    world = FakeWorld(obj=(-2.0, 0.3))

    async def no_backup(name, args, _orig=world.call_tool_prefixed):
        if name == "nav2__backup_robot":
            world.calls.append((name, args))
            return "Error: Backup operation failed"
        return await _orig(name, args)

    world.call_tool_prefixed = no_backup
    result = _run(world)
    assert result["success"], result["reason"]
    assert world.names()[:4].count("nav2__backup_robot") == 1
    assert "nav2__spin_robot" in world.names()
