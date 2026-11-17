"""Tests for the pick skill against a fake MCP client.

The fake answers with the texts the real servers send: the moveit MCP's
completed/failed/refused motions, compute_ik and joint-state listings,
the perception grasp pose and the GripperCommand action result.
"""

import asyncio
import json

from skill_based.skills import pick
from skill_based.utils import FLOOR_BOX, gripper_position, move_arm_to_look_forward

DONE = (
    "Plan and execute operation submitted. Operation ID: op1\n\n--- Execution Result ---\n"
    "Operation: op1\nStatus: completed\nProgress: 100.0%\nExecution completed successfully!\n"
)
REFUSED = "Error: execution op0 is still running; stop it or wait."

GRASP = {
    "centroid_base_frame": {"x": 0.95, "y": -0.07, "z": 0.05},
    "grasp_pose": {
        "frame_id": "base_footprint",
        "position": {"x": 0.95, "y": -0.07, "z": 0.24},
        "orientation": {"x": 0.72, "y": -0.69, "z": 0.0, "w": 0.0},
    },
    "pre_grasp_pose": {
        "frame_id": "base_footprint",
        "position": {"x": 0.95, "y": -0.07, "z": 0.40},
        "orientation": {"x": 0.72, "y": -0.69, "z": 0.0, "w": 0.0},
    },
    "bbox_base_frame": {"z_min": 0.004, "z_max": 0.074},
    "gripper_offset_m": 0.188,
    "grasp_xy_method": "near_face",
    "grasp_z_method": "centroid",
    "oriented": True,
}


def _joints_text(wrist_3: float) -> str:
    names = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
    values = [0.0, -2.3, -1.4, -0.9, 1.57, wrist_3]
    return "\n".join(f"  arm_{n}_joint: {v} rad" for n, v in zip(names, values))


class FakeMCP:
    def __init__(self, close_position=0.41, close_stalled=True, motion=DONE, wrist_for=None, finger=0.005,
                 open_position=0.003, close_raises=False, close_result=None, finger_after_close=0.41, grasp=GRASP,
                 box_reply="Successfully added collision box 'floor' to planning scene."):
        self.box_reply = box_reply
        self.finger = finger
        self.open_position = open_position
        self.close_raises = close_raises
        self.close_result = close_result
        self.finger_after_close = finger_after_close  # None: /joint_states without finger_joint
        self.grasp = grasp
        self.closed = False
        self.close_position = close_position
        self.close_stalled = close_stalled
        self.motion = motion
        self.wrist_for = wrist_for or (lambda q: 0.0)
        self.calls = []

    async def call_tool_prefixed(self, name, args):
        self.calls.append((name, args))
        if name == "moveit__plan_and_execute":
            return self.motion
        if name == "moveit__clear_planning_scene":
            return "Successfully cleared all collision objects from planning scene."
        if name == "moveit__add_collision_box":
            return self.box_reply
        if name == "moveit__compute_ik":
            return "Inverse kinematics solution for 'arm':\n" + _joints_text(self.wrist_for(args["orientation"]))
        if name == "moveit__get_current_joint_state":
            return "Current joint state for 'arm':\n" + _joints_text(0.0)
        if name == "ros__subscribe_once":
            finger = self.finger_after_close if self.closed else self.finger
            if finger is None:
                return json.dumps({"msg": {"name": [], "position": []}})
            return json.dumps({"msg": {"name": ["finger_joint"], "position": [finger]}})
        if name == "perception__segment_objects":
            return json.dumps({"status": "SUCCESS"})
        if name == "perception__get_topdown_grasp_pose":
            return json.dumps(self.grasp)
        if name == "ros__send_action_goal":
            target = args["goal"]["command"]["position"]
            if target == 0.0:
                result = {"position": self.open_position, "stalled": False, "reached_goal": True}
                return json.dumps({"status": 4, "result": result})
            self.closed = True
            if self.close_raises:
                raise RuntimeError("action server not available")
            if self.close_result is not None:
                return json.dumps({"status": 4, "result": self.close_result})
            result = {"position": self.close_position, "stalled": self.close_stalled, "reached_goal": False}
            return json.dumps({"status": 6 if self.close_stalled else 4, "result": result})
        return "ok"

    def names(self):
        return [n for n, _ in self.calls]


def _run(mcp):
    return asyncio.run(pick.run(mcp, "green cube"))


def test_pick_succeeds_when_the_gripper_stalls_on_the_object():
    mcp = FakeMCP()
    result = _run(mcp)
    assert result["success"], result["reason"]
    assert abs(result["held_object_height_m"] - 0.070) < 1e-6
    assert abs(result["pads_above_bottom_m"] - (0.24 - 0.188 - 0.004)) < 1e-6


def test_pre_grasp_height_comes_from_the_tool():
    mcp = FakeMCP()
    _run(mcp)
    ik_heights = [a["position"][2] for n, a in mcp.calls if n == "moveit__compute_ik"]
    assert 0.40 in ik_heights
    assert 0.24 + pick.PRE_GRASP_CLEARANCE_M not in ik_heights


def test_empty_grasp_fails_and_reopens_the_gripper():
    mcp = FakeMCP(close_position=0.70, close_stalled=False)
    result = _run(mcp)
    assert not result["success"]
    assert "empty grasp" in result["reason"]
    goals = [a["goal"]["command"]["position"] for n, a in mcp.calls if n == "ros__send_action_goal"]
    assert goals[-1] == 0.0


def test_floor_is_put_back_after_every_scene_clear():
    mcp = FakeMCP()
    _run(mcp)
    names = mcp.names()
    assert "moveit__clear_planning_scene" in names
    for i, name in enumerate(names):
        if name == "moveit__clear_planning_scene":
            assert names[i + 1] == "moveit__add_collision_box"
            assert mcp.calls[i + 1][1] == FLOOR_BOX


def test_refused_motion_is_not_success():
    mcp = FakeMCP(motion=REFUSED)
    result = _run(mcp)
    assert not result["success"]
    assert not any(n == "ros__send_action_goal" and a["goal"]["command"]["position"] > 0 for n, a in mcp.calls)


def test_look_forward_refused_is_not_success():
    result = asyncio.run(move_arm_to_look_forward(FakeMCP(motion=REFUSED)))
    assert not result["success"]


def test_mirrored_orientation_chosen_when_it_turns_the_wrist_less():
    # The tool's orientation needs a 3 rad wrist turn, its mirror 0.1 rad.
    q_tool = [0.72, -0.69, 0.0, 0.0]
    mcp = FakeMCP(wrist_for=lambda q: 3.0 if q == q_tool else 0.1)
    _run(mcp)
    plan_iks = [a["orientation"] for n, a in mcp.calls if n == "moveit__compute_ik"][2:]
    assert all(q == pick._mirror_about_tool_z(q_tool) for q in plan_iks)


def test_mirror_is_the_same_grip_turned_half_way():
    q = [0.882554, 0.47021, 0.0, 0.0]
    assert pick._mirror_about_tool_z(q) == [0.47021, -0.882554, 0.0, -0.0]


def test_refuses_to_start_with_a_closed_gripper():
    mcp = FakeMCP(finger=0.41)
    result = _run(mcp)
    assert not result["success"]
    assert "not open" in result["reason"]
    assert "moveit__plan_and_execute" not in mcp.names()


def test_wrist_turn_compared_across_the_wrap():
    # Current wrist 0.0; the tool's orientation lands at 6.2 rad (~ -0.08 after the wrap),
    # its mirror at 1.0 rad: the tool's orientation is the smaller turn.
    q_tool = [0.72, -0.69, 0.0, 0.0]
    mcp = FakeMCP(wrist_for=lambda q: 6.2 if q == q_tool else 1.0)
    _run(mcp)
    plan_iks = [a["orientation"] for n, a in mcp.calls if n == "moveit__compute_ik"][2:]
    assert all(q == q_tool for q in plan_iks)


def test_failed_lift_is_a_failed_pick_and_the_arm_stays():
    mcp = FakeMCP()
    plans = []

    async def call(name, args, _orig=mcp.call_tool_prefixed):
        if name == "moveit__plan_and_execute":
            plans.append(args)
            # look_forward, pre-grasp, descend succeed; the lift and its retry are refused
            if len(plans) > 3:
                mcp.calls.append((name, args))
                return REFUSED
        return await _orig(name, args)

    mcp.call_tool_prefixed = call
    result = _run(mcp)
    assert not result["success"]
    assert "lift failed" in result["reason"]
    assert len(plans) == 5


def _motions_after_close(mcp):
    i = max(i for i, (n, a) in enumerate(mcp.calls)
            if n == "ros__send_action_goal" and a["goal"]["command"]["position"] > 0)
    return [n for n, _ in mcp.calls[i + 1:] if n in ("moveit__plan_and_execute", "ros__send_action_goal")]


def test_open_not_confirmed_stops_before_the_descent():
    mcp = FakeMCP(open_position=0.30)
    result = _run(mcp)
    assert not result["success"]
    assert "did not open" in result["reason"]
    assert mcp.names().count("moveit__plan_and_execute") == 1  # look_forward only


def test_close_error_is_unknown_and_the_arm_stays():
    mcp = FakeMCP(close_raises=True, finger_after_close=0.41)
    result = _run(mcp)
    assert "unknown" in result["reason"]
    assert _motions_after_close(mcp) == []


def test_close_without_a_position_is_unknown_not_empty():
    mcp = FakeMCP(close_result={"stalled": False}, finger_after_close=None)
    result = _run(mcp)
    assert "unknown" in result["reason"]
    assert _motions_after_close(mcp) == []


def test_close_error_with_a_fully_closed_reread_is_empty():
    mcp = FakeMCP(close_raises=True, finger_after_close=0.70)
    assert "empty grasp" in _run(mcp)["reason"]


def test_malformed_grasp_pose_fails_cleanly():
    no_orientation = {k: v for k, v in GRASP["grasp_pose"].items() if k != "orientation"}
    for grasp in ({**GRASP, "pre_grasp_pose": {"frame_id": "base_footprint"}},
                  {**GRASP, "grasp_pose": no_orientation}):
        mcp = FakeMCP(grasp=grasp)
        result = _run(mcp)
        assert not result["success"]
        assert "malformed grasp pose" in result["reason"]
        assert mcp.names().count("moveit__plan_and_execute") == 1


def test_failed_retract_after_a_failed_descent_is_reported():
    mcp = FakeMCP()
    plans = []

    async def call(name, args, _orig=mcp.call_tool_prefixed):
        if name == "moveit__plan_and_execute":
            plans.append(args)
            # look_forward and pre-grasp succeed; descent, its retry and the retract are refused
            if len(plans) > 2:
                mcp.calls.append((name, args))
                return REFUSED
        return await _orig(name, args)

    mcp.call_tool_prefixed = call
    result = _run(mcp)
    assert "descent plan failed" in result["reason"]
    assert "retract failed too" in result["reason"]


def test_nan_finger_reread_is_unknown_not_empty():
    mcp = FakeMCP(close_raises=True, finger_after_close=float("nan"))
    result = _run(mcp)
    assert "unknown" in result["reason"]
    assert _motions_after_close(mcp) == []


def test_non_finite_grasp_pose_fails_before_any_motion():
    pose = {**GRASP["grasp_pose"], "position": {"x": float("nan"), "y": -0.07, "z": 0.24}}
    mcp = FakeMCP(grasp={**GRASP, "grasp_pose": pose})
    result = _run(mcp)
    assert "malformed grasp pose" in result["reason"]
    assert mcp.names().count("moveit__plan_and_execute") == 1  # look_forward only


def test_malformed_bbox_leaves_the_height_unknown():
    mcp = FakeMCP(grasp={**GRASP, "bbox_base_frame": {"z_min": "n/a", "z_max": 0.074}})
    result = _run(mcp)
    assert result["success"], result["reason"]
    assert result["held_object_height_m"] is None


def test_floor_not_restored_stops_the_pick():
    mcp = FakeMCP(box_reply="Error: planning scene monitor not available")
    result = _run(mcp)
    assert not result["success"]
    assert "scene reset failed" in result["reason"]
    assert mcp.names().count("moveit__plan_and_execute") == 1  # look_forward only


def test_refused_pre_grasp_is_not_success():
    mcp = FakeMCP()
    plans = []

    async def call(name, args, _orig=mcp.call_tool_prefixed):
        if name == "moveit__plan_and_execute":
            plans.append(args)
            # look_forward succeeds; the pre-grasp and its retry are refused
            if len(plans) > 1:
                mcp.calls.append((name, args))
                return REFUSED
        return await _orig(name, args)

    mcp.call_tool_prefixed = call
    result = _run(mcp)
    assert not result["success"]
    assert "pre-grasp plan failed" in result["reason"]
    assert not any(n == "ros__send_action_goal" and a["goal"]["command"]["position"] > 0 for n, a in mcp.calls)


def test_gripper_position_retries_a_short_message():
    replies = iter([
        {"msg": {"name": ["finger_joint"], "position": []}},
        {"msg": {"name": ["finger_joint"], "position": [0.41]}},
    ])

    class Joints:
        async def call_tool_prefixed(self, name, args):
            return json.dumps(next(replies))

    assert asyncio.run(gripper_position(Joints())) == 0.41
