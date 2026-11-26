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
NAV_FAILED = json.dumps({"error": "NavigationError", "message": "Navigation to pose failed",
                         "error_code": "NAVIGATION_FAILED"})
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
        # nav2 navigate_to_pose: reaches its goal this far short and this far off its heading (rad),
        # or gives up (no path) after nav_fail_after metres, or the call raises nav_raise
        self.nav_short = 0.15
        self.nav_yaw_err = math.radians(10)
        self.nav_fail_after = None
        self.nav_raise = None

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
        if name == "nav2__get_robot_pose":
            return json.dumps({"position": {"x": self.x, "y": self.y}, "orientation": {"yaw": self.yaw}})
        if name == "nav2__navigate_to_pose":
            if self.nav_raise:
                raise self.nav_raise
            gx, gy = args["x"], args["y"]
            d = math.hypot(gx - self.x, gy - self.y)
            moved = self.nav_fail_after if self.nav_fail_after is not None else max(0.0, d - self.nav_short)
            self.x += (gx - self.x) * moved / d
            self.y += (gy - self.y) * moved / d
            self.yaw = args["yaw"] + self.nav_yaw_err
            if self.nav_fail_after is not None:
                return NAV_FAILED
            return f"Successfully navigated to pose ({gx:.2f}, {gy:.2f}, {args['yaw']:.2f})"
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


@pytest.fixture
def no_nav(monkeypatch):
    """Drive the whole way in legs, as before nav2 was used for the long way."""
    monkeypatch.setattr(approach, "NAV_ABOVE_M", 99.0)


def _run(world, next_action="pick"):
    return asyncio.run(approach.run(world, world.name, next_action))


def _at_standoff(world, next_action="pick"):
    return approach._in_window(world.near_face(), next_action)


def test_object_straight_ahead_ends_at_the_standoff():
    world = FakeWorld(obj=(1.3, 0.0))
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
    assert result["failure"] == "not_found"
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
    assert result["failure"] == "turn_failed"
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


def test_lost_close_to_the_standoff_drives_the_rest_blind(no_nav):
    world = FakeWorld(obj=(1.9 + HALF_DEPTH, 0.0))  # first leg 0.8 m leaves 0.4 m to go
    world.blind_below = 1.2
    result = _run(world, "surface_place")
    assert result["success"], result["reason"]
    assert result["verified"] is False
    assert world.names().count("nav2__drive_on_heading") == 2
    assert _at_standoff(world, "surface_place")


def test_object_lost_far_from_the_standoff_still_fails(no_nav):
    world = FakeWorld(obj=(3.5, 0.0))  # first leg 0.8 m ends ~1.8 m short of the standoff
    world.blind_below = 3.0
    result = _run(world)
    assert not result["success"]
    assert result["failure"] == "lost"
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


def test_without_nav2_the_long_way_is_driven_in_short_legs(no_nav):
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


def test_perception_down_is_not_reported_as_not_found():
    world = FakeWorld(obj=(2.0, 0.0))

    async def sam3_down(name, args, _orig=world.call_tool_prefixed):
        if name == "perception__segment_objects":
            world.calls.append((name, args))
            return "ERROR: Connection failed"
        return await _orig(name, args)

    world.call_tool_prefixed = sam3_down
    result = _run(world)
    assert not result["success"]
    assert result["failure"] == "perception_error"  # approach_15d: SAM3 unreachable
    assert "nav2__drive_on_heading" not in world.names()


def test_arm_failure_names_the_moveit_verdict():
    world = FakeWorld()

    async def arm_fails(name, args, _orig=world.call_tool_prefixed):
        if name == "moveit__plan_and_execute":
            world.calls.append((name, args))
            return ("Plan and execute operation submitted. Operation ID: 9d74\n"
                    "--- Execution Result ---\nOperation: 9d74\nStatus: completed\nExecution failed.")
        return await _orig(name, args)

    world.call_tool_prefixed = arm_fails
    result = _run(world)
    assert result["failure"] == "arm_failed"
    assert "Execution failed" in result["reason"] and "Operation ID" not in result["reason"]
    assert "The base did not move." in result["reason"]


def test_refused_drive_is_reported_with_where_the_object_is(no_nav):
    world = FakeWorld(obj=(2.0, 0.0))

    async def blocked(name, args, _orig=world.call_tool_prefixed):
        if name == "nav2__drive_on_heading":
            world.calls.append((name, args))
            return json.dumps({"error": "NavigationError", "message": "Drive on heading operation failed",
                               "error_code": "NAVIGATION_FAILED"}, indent=2)
        return await _orig(name, args)

    world.call_tool_prefixed = blocked
    world.nav_fail_after = 0.0  # no way around either
    result = _run(world)
    assert result["failure"] == "nav_failed"
    assert "1.96 m" in result["reason"]  # where it saw the object
    assert "did not move" not in result["reason"]  # approach_24: it drove until the obstacle
    assert "stopped partway through a 0.80 m drive" in result["reason"]


def test_failure_carries_the_steps_and_success_does_not():
    world = FakeWorld(name="green cube")
    failed = asyncio.run(approach.run(world, "red ball", "pick"))
    assert any("backup 0.40 m" in line for line in failed["steps"])
    assert "turned 23 times" in failed["reason"]
    world = FakeWorld(obj=(2.0, 0.0))
    assert "steps" not in _run(world)


def test_search_turns_two_rounds_the_second_shifted_half_a_step():
    world = FakeWorld(name="green cube")
    result = asyncio.run(approach.run(world, "red ball", "pick"))
    assert result["failure"] == "not_found"
    turns = [round(math.degrees(a["angle"]), 1) for n, a in world.calls if n == "nav2__spin_robot"]
    assert turns == [30.0] * 11 + [45.0] + [30.0] * 11  # run 04: 60 deg steps missed the container


def test_missing_point_cloud_is_segmented_again():
    world = FakeWorld(obj=(2.0, 0.0))
    first = [True]

    async def no_cloud_once(name, args, _orig=world.call_tool_prefixed):
        if name == "perception__segment_objects" and first[0]:
            first[0] = False
            world.calls.append((name, args))
            return ("ERROR: Segmentation service reported SUCCESS, but its point cloud did not arrive "
                    "within 20s. Please segment again.")
        return await _orig(name, args)

    world.call_tool_prefixed = no_cloud_once
    result = _run(world)
    assert result["success"], result["reason"]
    assert "nav2__backup_robot" not in world.names()  # found on the first look, after the retry
    assert [a["camera"] for n, a in world.calls if n == "perception__segment_objects"][:2] == ["front", "front"]


def test_motion_is_told_in_time_order():
    world = FakeWorld(obj=(-2.0, 0.3))

    async def no_backup(name, args, _orig=world.call_tool_prefixed):
        if name == "nav2__backup_robot":
            world.calls.append((name, args))
            return "Error: Backup operation failed"
        return await _orig(name, args)

    world.call_tool_prefixed = no_backup
    result = _run(world)
    reason = result["reason"]
    # run 01: the refused backup came first but was reported last
    assert reason.index("stopped partway through a 0.40 m backup") < reason.index("turned")
    assert reason.index("turned") < reason.index("drove")


def test_far_object_goes_with_nav2_then_short_legs():
    world = FakeWorld(obj=(3.5, 0.4))
    result = _run(world, "container_place")
    assert result["success"], result["reason"]
    assert _at_standoff(world, "container_place")
    names = world.names()
    assert names.count("nav2__navigate_to_pose") == 1
    assert names.index("nav2__navigate_to_pose") < names.index("nav2__drive_on_heading")
    assert "with nav2" in result["reason"]


def test_near_object_does_not_use_nav2():
    world = FakeWorld(obj=(1.3, 0.0))  # near face 1.26 m: within NAV_ABOVE_M of the 0.85 m standoff
    _run(world)
    assert "nav2__navigate_to_pose" not in world.names()


def test_nav2_goal_stays_short_of_the_standoff_facing_the_object():
    world = FakeWorld(obj=(3.0 + HALF_DEPTH, 0.0))
    _run(world, "container_place")
    goal = next(a for n, a in world.calls if n == "nav2__navigate_to_pose")
    # near face 3.0 m, standoff 0.65 m: goal NAV_SHORT_M before the standoff, facing it
    assert goal["x"] == pytest.approx(3.0 - 0.65 - approach.NAV_SHORT_M, abs=0.01)
    assert goal["y"] == pytest.approx(0.0, abs=0.01) and goal["yaw"] == pytest.approx(0.0, abs=0.01)


def test_nav2_without_a_path_is_nav_failed_with_the_real_distance():
    world = FakeWorld(obj=(3.5, 0.0))
    world.nav_fail_after = 1.1
    result = _run(world)
    assert result["failure"] == "nav_failed"
    assert "Navigation to pose failed" in result["reason"]
    assert "drove 1.10 m with nav2" in result["reason"]  # taken from the map pose, not the goal
    assert "nav2__drive_on_heading" not in world.names()


def test_nav2_timeout_cancels_the_goal():
    world = FakeWorld(obj=(3.5, 0.0))
    world.nav_raise = TimeoutError("read timeout")
    result = _run(world)
    assert result["failure"] == "nav_failed"
    assert "nav2__cancel_navigation" in world.names()


def test_off_heading_after_nav2_turns_once_and_finds_it():
    world = FakeWorld(obj=(3.5, 0.0))
    world.nav_yaw_err = math.radians(40)  # beyond both cameras' view
    result = _run(world)
    assert result["success"], result["reason"]
    names = world.names()
    after = names[names.index("nav2__navigate_to_pose"):]
    assert after.index("nav2__spin_robot") < after.index("nav2__drive_on_heading")


def test_refused_straight_drive_is_driven_by_nav2_instead(no_nav):
    world = FakeWorld(obj=(1.6, 0.0))
    refused = [True]

    async def blocked_once(name, args, _orig=world.call_tool_prefixed):
        if name == "nav2__drive_on_heading" and refused[0]:
            refused[0] = False
            world.calls.append((name, args))
            return "Drive on heading operation failed"
        return await _orig(name, args)

    world.call_tool_prefixed = blocked_once
    world.nav_short = 0.0
    world.nav_yaw_err = 0.0
    result = _run(world)  # approach_4: nav2 stopped beside the bin, the straight line to the box was blocked
    assert result["success"], result["reason"]
    goal = next(a for n, a in world.calls if n == "nav2__navigate_to_pose")
    assert goal["x"] == pytest.approx(1.6 - HALF_DEPTH - approach.STANDOFF_BY_NEXT_ACTION["pick"], abs=0.02)
    assert "stopped partway" in result["reason"] and "with nav2" in result["reason"]


def test_refused_backup_is_not_driven_by_nav2():
    world = FakeWorld(obj=(0.5 + HALF_DEPTH, 0.0))  # near face 0.5 m, floor_place wants 0.85: backs up

    async def no_backup(name, args, _orig=world.call_tool_prefixed):
        if name == "nav2__backup_robot":
            world.calls.append((name, args))
            return "Backup operation failed"
        return await _orig(name, args)

    world.call_tool_prefixed = no_backup
    result = _run(world, "floor_place")
    assert result["failure"] == "drive_failed"
    assert "nav2__navigate_to_pose" not in world.names()  # it would have driven towards the object


def test_nav2_goal_geometry_off_axis():
    m = {"dist": 2.04, "near": 2.0, "bearing": math.radians(90)}
    goal = approach._nav_goal((1.0, 1.0, 0.0), m, standoff=0.65, short=0.3)
    assert goal["x"] == pytest.approx(1.0, abs=1e-6) and goal["y"] == pytest.approx(1.0 + 2.0 - 0.95)
    assert goal["yaw"] == pytest.approx(math.pi / 2) and goal["obj"][1] == pytest.approx(3.04)


def test_drive_refused_after_reaching_the_standoff_is_not_sent_to_nav2(no_nav):
    world = FakeWorld(obj=(1.3, 0.0))  # near face 1.26 m, one 0.41 m leg to the 0.85 m standoff

    async def stops_late(name, args, _orig=world.call_tool_prefixed):
        if name == "nav2__drive_on_heading":
            world.calls.append((name, args))
            world.x += args["distance"] + 0.05  # it overran a little, then reported a failure
            return "Drive on heading operation failed"
        return await _orig(name, args)

    world.call_tool_prefixed = stops_late
    result = _run(world)
    assert result["success"], result["reason"]
    assert "nav2__navigate_to_pose" not in world.names()  # a goal behind the robot
