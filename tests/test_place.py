"""Tests for the place skill's container and surface modes against a fake MCP client."""

import asyncio
import json

import pytest

from skill_based.skills import place
from test_pick import DONE, REFUSED, FakeMCP as _PickFake

BIN = {
    "centroid_base_frame": {"x": 0.90, "y": 0.05, "z": 0.22},
    "grasp_pose": {
        "frame_id": "base_footprint",
        "position": {"x": 0.88, "y": 0.04, "z": 0.60},
        "orientation": {"x": 1.0, "y": 0.0, "z": 0.0, "w": 0.0},
    },
    "bbox_base_frame": {"z_min": 0.0, "z_max": 0.45},  # container needs only the rim
}

# The grey box as a table: near face 0.75 m ahead, top 0.30 m.
BOX = {
    "grasp_pose": {
        "frame_id": "base_footprint",
        "position": {"x": 0.90, "y": 0.02, "z": 0.50},
        "orientation": {"x": 1.0, "y": 0.0, "z": 0.0, "w": 0.0},
    },
    "bbox_base_frame": {"x_min": 0.75, "x_max": 1.05, "y_min": -0.18, "y_max": 0.22, "z_min": 0.0, "z_max": 0.30},
}


class FakeMCP(_PickFake):
    def __init__(self, bin_pose=BIN, open_position=0.003, seg=None, refuse_drop=False, box_error=False,
                 drop_reply=None, **kw):
        kw.setdefault("finger", 0.41)  # place starts with the object held
        super().__init__(**kw)
        self.box_error = box_error
        self.refuse_drop = refuse_drop
        self.drop_reply = drop_reply
        self.bin_pose = bin_pose
        self.open_position = open_position
        self.seg = seg or (lambda prompt, camera: "SUCCESS")

    async def call_tool_prefixed(self, name, args):
        if name == "moveit__plan_and_execute" and self.refuse_drop and "moveit__compute_ik" in self.names():
            self.calls.append((name, args))
            return REFUSED
        if name == "moveit__plan_and_execute" and self.drop_reply and "moveit__compute_ik" in self.names():
            self.calls.append((name, args))
            return self.drop_reply
        if name == "moveit__add_collision_box" and self.box_error and args["object_id"] == "container":
            self.calls.append((name, args))
            raise RuntimeError("scene service unavailable")
        if name == "perception__segment_objects":
            self.calls.append((name, args))
            reply = self.seg(args["prompt"], args["camera"])
            return json.dumps(reply if isinstance(reply, dict) else {"status": reply})
        if name == "perception__get_topdown_grasp_pose":
            self.calls.append((name, args))
            return json.dumps(self.bin_pose)
        if name == "ros__send_action_goal" and args["goal"]["command"]["position"] == 0.0:
            self.calls.append((name, args))
            result = {"position": self.open_position, "stalled": False, "reached_goal": True}
            return json.dumps({"status": 4, "result": result})
        return await super().call_tool_prefixed(name, args)


@pytest.fixture(autouse=True)
def no_settle_wait(monkeypatch):
    async def no_sleep(_):
        return None

    monkeypatch.setattr(place.asyncio, "sleep", no_sleep)


def _run(mcp, mode="container", height=0.06, pads=None):
    return asyncio.run(place.run(mcp, "trash bin", "green cube", mode, height, pads))


def _box(mcp):
    return next(a for n, a in mcp.calls if n == "moveit__add_collision_box" and a["object_id"] != "floor")


def _is_open(name, args):
    return name == "ros__send_action_goal" and args["goal"]["command"]["position"] == 0.0


def _release_index(mcp):
    return next((i for i, (n, a) in enumerate(mcp.calls) if _is_open(n, a)), len(mcp.calls))


def _drop_heights(mcp):
    """IK heights up to the release (the lift after it is not part of the drop)."""
    return [a["position"][2] for n, a in mcp.calls[:_release_index(mcp)] if n == "moveit__compute_ik"]


def _after_release(mcp):
    return mcp.calls[_release_index(mcp) + 1:]


def test_container_drop_succeeds_above_the_rim():
    mcp = FakeMCP()
    result = _run(mcp)
    assert result["success"], result["reason"]
    assert abs(_drop_heights(mcp)[0] - (0.45 + place.CONTAINER_RIM_CLEARANCE_M)) < 1e-9


def test_bin_box_is_added_after_the_scene_reset_and_kept_on_retry():
    mcp = FakeMCP(refuse_drop=True)
    result = _run(mcp)
    assert not result["success"]
    names = mcp.names()
    assert names.count("moveit__compute_ik") == 2
    box = [i for i, (n, a) in enumerate(mcp.calls) if n == "moveit__add_collision_box" and a["object_id"] == "container"]
    assert len(box) == 1
    assert "moveit__clear_planning_scene" not in names[box[0]:]


def test_out_of_reach_bin_fails_before_any_drop_motion():
    far = json.loads(json.dumps(BIN))
    far["grasp_pose"]["position"]["x"] = 1.20
    mcp = FakeMCP(bin_pose=far)
    result = _run(mcp)
    assert not result["success"]
    assert "Navigator" in result["reason"]
    assert not any(n == "ros__send_action_goal" for n in mcp.names())


def test_gripper_that_stays_closed_is_a_failure():
    mcp = FakeMCP(open_position=0.41)
    result = _run(mcp)
    assert not result["success"]
    assert "did not open" in result["reason"]


def test_object_not_seen_after_release_is_a_failure():
    mcp = FakeMCP(seg=lambda prompt, camera: "NO_OBJECTS_FOUND" if prompt == "green cube" else "SUCCESS")
    result = _run(mcp)
    assert not result["success"]
    assert "verify" in result["reason"]


NO_CLOUD = {"error": "Segmentation of 'green cube' on arm camera reported SUCCESS, but its point cloud did not arrive. Nothing is cached; segment again."}
TIMEOUT = {"error": "Timeout (20s) waiting for segmentation result on /seg_status."}


def test_object_found_without_its_cloud_counts_as_placed():
    mcp = FakeMCP(seg=lambda prompt, camera: NO_CLOUD if prompt == "green cube" else "SUCCESS")
    result = _run(mcp)
    assert result["success"], result["reason"]


def test_segmentation_error_is_reported_as_unverified_not_missed():
    mcp = FakeMCP(seg=lambda prompt, camera: TIMEOUT if prompt == "green cube" else "SUCCESS")
    result = _run(mcp)
    assert not result["success"]
    assert "could not verify" in result["reason"]
    assert "missed" not in result["reason"]


def test_bin_falls_back_to_the_front_camera():
    mcp = FakeMCP(seg=lambda prompt, camera: "SUCCESS" if camera == "front" or prompt == "green cube" else "NO_OBJECTS_FOUND")
    result = _run(mcp)
    assert result["success"], result["reason"]


# The orange cylinder as the reference on the floor, 1.05 m straight ahead.
CYLINDER = {
    "grasp_pose": {
        "frame_id": "base_footprint",
        "position": {"x": 1.05, "y": 0.0, "z": 0.30},
        "orientation": {"x": 1.0, "y": 0.0, "z": 0.0, "w": 0.0},
    },
    "bbox_base_frame": {"x_min": 1.01, "x_max": 1.09, "y_min": -0.04, "y_max": 0.04, "z_min": 0.0, "z_max": 0.12},
}


def _ik_points(mcp):
    return [tuple(a["position"]) for n, a in mcp.calls if n == "moveit__compute_ik"]


def _ref_at(x, y, half=0.04):
    pose = json.loads(json.dumps(CYLINDER))
    pose["grasp_pose"]["position"].update({"x": x, "y": y})
    pose["bbox_base_frame"].update({"x_min": x - half, "x_max": x + half, "y_min": y - half, "y_max": y + half})
    return pose


def test_floor_sets_the_object_down_in_front_of_the_reference():
    mcp = FakeMCP(bin_pose=CYLINDER)
    result = _run(mcp, "floor", height=0.066, pads=0.048)
    assert result["success"], result["reason"]
    assert "next to" in result["reason"]
    final = 0.0 + 0.005 + 0.048 + 0.188
    assert _drop_heights(mcp) == pytest.approx([final + 0.08, final])
    x, y, _ = _ik_points(mcp)[-1]
    # Near face at x_min 1.01: 13 cm in front of it.
    assert (x, y) == pytest.approx((1.01 - 0.13, 0.0))


def test_floor_offset_points_back_towards_the_robot():
    mcp = FakeMCP(bin_pose=_ref_at(0.9, 0.3))
    _run(mcp, "floor", height=0.066, pads=0.048)
    x, y, _ = _ik_points(mcp)[-1]
    ref = (0.9 ** 2 + 0.3 ** 2) ** 0.5
    assert y / x == pytest.approx(0.3 / 0.9)
    assert ref - (x ** 2 + y ** 2) ** 0.5 > 0.13


def test_floor_next_to_a_big_box_stays_clear_of_its_front():
    mcp = FakeMCP(bin_pose=_ref_at(1.0, 0.0, half=0.15))
    result = _run(mcp, "floor", height=0.066, pads=0.048)
    assert result["success"], result["reason"]
    x, _, _ = _ik_points(mcp)[-1]
    assert x == pytest.approx(0.85 - 0.13)


def test_floor_reference_goes_into_the_scene():
    mcp = FakeMCP(bin_pose=CYLINDER)
    _run(mcp, "floor", height=0.066, pads=0.048)
    box = _box(mcp)
    assert box["object_id"] == "reference"
    assert box["position"] == pytest.approx([1.05, 0.0, 0.06])
    assert box["dimensions"] == pytest.approx([0.08, 0.08, 0.12])


@pytest.mark.parametrize("ref_x", [0.62, 1.25])
def test_floor_drop_outside_the_reach_band_stops_before_motion(ref_x):
    mcp = FakeMCP(bin_pose=_ref_at(ref_x, 0.0))
    result = _run(mcp, "floor", height=0.066, pads=0.048)
    assert not result["success"]
    assert "moveit__compute_ik" not in mcp.names()


def test_floor_mode_keeps_the_scene_floor_box():
    mcp = FakeMCP(bin_pose=CYLINDER)
    _run(mcp, "floor", height=0.066, pads=0.048)
    floor_boxes = [a for n, a in mcp.calls if n == "moveit__add_collision_box" and a["object_id"] == "floor"]
    assert len(floor_boxes) == 1  # only the one reset_scene adds


def test_floor_without_a_height_stops_before_any_motion():
    mcp = FakeMCP(bin_pose=CYLINDER)
    result = _run(mcp, "floor", height=0.0)
    assert not result["success"]
    assert "held_object_height_m" in result["reason"]
    assert mcp.calls == []


def test_surface_drop_puts_the_object_bottom_above_the_top():
    mcp = FakeMCP(bin_pose=BOX)
    result = _run(mcp, "surface", height=0.065)
    assert result["success"], result["reason"]
    assert "onto" in result["reason"]
    assert _drop_heights(mcp)[-1] == pytest.approx(0.30 + 0.03 + 0.065 + 0.188)


def test_surface_with_measured_pads_sets_the_object_down():
    mcp = FakeMCP(bin_pose=BOX)
    result = _run(mcp, "surface", height=0.066, pads=0.048)
    assert result["success"], result["reason"]
    final = 0.30 + 0.005 + 0.048 + 0.188
    assert _drop_heights(mcp) == pytest.approx([final + 0.08, final])


def test_implausible_pads_fall_back_to_the_worst_case():
    mcp = FakeMCP(bin_pose=BOX)
    _run(mcp, "surface", height=0.066, pads=0.20)
    assert _drop_heights(mcp)[-1] == pytest.approx(0.30 + 0.03 + 0.066 + 0.188)


def test_container_drop_is_one_motion():
    mcp = FakeMCP()
    _run(mcp, pads=0.048)
    assert len(_drop_heights(mcp)) == 1


def test_surface_box_top_stays_below_the_held_object():
    mcp = FakeMCP(bin_pose=BOX)
    _run(mcp, "surface", height=0.065)
    box = _box(mcp)
    box_top = box["position"][2] + box["dimensions"][2] / 2
    object_bottom = _drop_heights(mcp)[-1] - 0.188 - 0.065
    assert object_bottom - box_top == pytest.approx(0.03)


def test_surface_box_is_the_measured_footprint():
    mcp = FakeMCP(bin_pose=BOX)
    _run(mcp, "surface", height=0.065)
    box = _box(mcp)
    assert box["object_id"] == "surface"
    assert box["position"] == pytest.approx([0.90, 0.02, 0.15])
    assert box["dimensions"] == pytest.approx([0.30, 0.40, 0.30])


def test_container_keeps_its_fixed_box():
    mcp = FakeMCP()
    _run(mcp)
    box = _box(mcp)
    assert box["object_id"] == "container"
    assert box["dimensions"] == pytest.approx([0.40, 0.40, 0.45])


def test_surface_without_a_height_stops_before_any_motion():
    mcp = FakeMCP(bin_pose=BOX)
    result = _run(mcp, "surface", height=0.0)
    assert not result["success"]
    assert "held_object_height_m" in result["reason"]
    assert mcp.calls == []


@pytest.mark.parametrize("top", [0.02, 0.95])
def test_implausible_surface_top_stops_before_the_drop(top):
    pose = json.loads(json.dumps(BOX))
    pose["bbox_base_frame"]["z_max"] = top
    mcp = FakeMCP(bin_pose=pose)
    result = _run(mcp, "surface", height=0.065)
    assert not result["success"]
    assert "outside" in result["reason"]
    assert "moveit__compute_ik" not in mcp.names()


def test_far_surface_gets_the_object_behind_its_near_edge():
    pose = json.loads(json.dumps(BOX))
    pose["grasp_pose"]["position"]["x"] = 1.20
    pose["bbox_base_frame"].update({"x_min": 0.75, "x_max": 1.65})
    mcp = FakeMCP(bin_pose=pose)
    result = _run(mcp, "surface", height=0.065)
    assert result["success"], result["reason"]
    x = next(a["position"][0] for n, a in mcp.calls if n == "moveit__compute_ik")
    assert x == pytest.approx(0.75 + 0.15)


def test_no_drop_without_the_bin_in_the_scene():
    mcp = FakeMCP(box_error=True)
    result = _run(mcp)
    assert not result["success"]
    assert "collision box" in result["reason"]
    assert "moveit__compute_ik" not in mcp.names()
    assert not any(n == "ros__send_action_goal" for n in mcp.names())


def test_no_drop_when_the_bin_box_is_refused_as_text():
    mcp = FakeMCP(box_reply="Error: planning scene monitor not available")
    # the floor box gets the same reply, so make only the container fail
    floor_ok = "Successfully added collision box 'floor' to planning scene."

    async def call(name, args, _orig=mcp.call_tool_prefixed):
        if name == "moveit__add_collision_box" and args["object_id"] == "floor":
            mcp.calls.append((name, args))
            return floor_ok
        return await _orig(name, args)

    mcp.call_tool_prefixed = call
    result = _run(mcp)
    assert not result["success"]
    assert "collision box" in result["reason"]
    assert "moveit__compute_ik" not in mcp.names()
    assert not any(n == "ros__send_action_goal" for n in mcp.names())


def test_floor_not_restored_stops_the_place():
    mcp = FakeMCP(box_reply="Error: planning scene monitor not available")
    result = _run(mcp)
    assert not result["success"]
    assert "scene reset failed" in result["reason"]
    assert "perception__segment_objects" not in mcp.names()


@pytest.mark.parametrize("finger", [0.003, 0.69, None], ids=["open", "closed_empty", "unknown"])
def test_nothing_held_stops_before_any_motion(finger):
    mcp = FakeMCP(finger=finger)
    result = _run(mcp)
    assert not result["success"]
    assert "nothing held" in result["reason"]
    assert not any(n.startswith("moveit__") for n, _ in mcp.calls)


def test_aborted_drop_motion_is_a_failure():
    # moveit-mcp reports a trajectory the controller aborted as completed + "Execution failed."
    aborted = DONE.replace("Execution completed successfully!", "Execution failed.")
    assert aborted != DONE
    mcp = FakeMCP(drop_reply=aborted)
    result = _run(mcp)
    assert not result["success"]
    assert "drop pose plan failed" in result["reason"]
    assert not any(n == "ros__send_action_goal" and a["goal"]["command"]["position"] == 0.0 for n, a in mcp.calls)


def test_unreadable_open_result_is_a_failure():
    mcp = FakeMCP(open_position=float("nan"))
    result = _run(mcp)
    assert not result["success"]
    assert "did not open" in result["reason"]


@pytest.mark.parametrize("mode,pose", [("floor", CYLINDER), ("surface", BOX)])
def test_arm_lifts_straight_up_before_the_verify(mode, pose):
    mcp = FakeMCP(bin_pose=pose)
    result = _run(mcp, mode, height=0.066, pads=0.048)
    assert result["success"], result["reason"]
    drop = _ik_points(mcp)[len(_drop_heights(mcp)) - 1]
    after = _after_release(mcp)
    lift = next(a for n, a in after if n == "moveit__compute_ik")["position"]
    assert lift == pytest.approx([drop[0], drop[1], drop[2] + place.VERIFY_LIFT_M])
    names = [n for n, _ in after]
    assert names.index("moveit__compute_ik") < names.index("perception__segment_objects")


def test_container_verifies_without_a_lift():
    mcp = FakeMCP()
    result = _run(mcp)
    assert result["success"], result["reason"]
    names = [n for n, _ in _after_release(mcp)]
    assert "moveit__compute_ik" not in names[:names.index("perception__segment_objects")]


def test_failed_lift_still_verifies():
    mcp = FakeMCP(bin_pose=CYLINDER)
    real = mcp.call_tool_prefixed

    async def refuse_lift(name, args):
        if name == "moveit__plan_and_execute" and any(_is_open(n, a) for n, a in mcp.calls):
            mcp.calls.append((name, args))
            return REFUSED
        return await real(name, args)

    mcp.call_tool_prefixed = refuse_lift
    result = _run(mcp, "floor", height=0.066, pads=0.048)
    assert result["success"], result["reason"]
    assert "perception__segment_objects" in [n for n, _ in _after_release(mcp)]


def test_surface_uses_the_robust_top_when_perception_has_it():
    box = json.loads(json.dumps(BOX))
    box["bbox_base_frame"]["z_max"] = 0.35  # a stray point 5 cm above the lid
    box["top_z"] = 0.30
    mcp = FakeMCP(bin_pose=box)
    result = _run(mcp, "surface", height=0.066, pads=0.048)
    assert result["success"], result["reason"]
    assert _drop_heights(mcp)[-1] == pytest.approx(0.30 + 0.005 + 0.048 + 0.188)
    assert _box(mcp)["dimensions"][2] == pytest.approx(0.30)


def test_container_keeps_the_highest_point_as_its_rim():
    rim = json.loads(json.dumps(BIN))
    rim["top_z"] = 0.40
    mcp = FakeMCP(bin_pose=rim)
    _run(mcp)
    assert _drop_heights(mcp)[0] == pytest.approx(0.45 + place.CONTAINER_RIM_CLEARANCE_M)


@pytest.mark.parametrize("x,y,half", [(0.9, 0.3, 0.04), (0.95, -0.35, 0.12), (1.0, 0.0, 0.15), (0.85, 0.5, 0.08)])
def test_floor_drop_keeps_the_gap_to_the_whole_reference_footprint(x, y, half):
    mcp = FakeMCP(bin_pose=_ref_at(x, y, half))
    result = _run(mcp, "floor", height=0.066, pads=0.048)
    assert result["success"], result["reason"]
    dx, dy, _ = _ik_points(mcp)[len(_drop_heights(mcp)) - 1]
    # Nearest point of the reference's footprint rectangle to the drop point.
    nx = min(max(dx, x - half), x + half)
    ny = min(max(dy, y - half), y + half)
    assert ((dx - nx) ** 2 + (dy - ny) ** 2) ** 0.5 >= place.FLOOR_GAP_M - 1e-9


def test_deep_grasp_keeps_the_fingertips_above_the_surface():
    # pick_4 on the robot: pads 3.5 cm above the cube's bottom, fingertips 3.7 cm below the pads
    mcp = FakeMCP(bin_pose=BOX)
    result = _run(mcp, "surface", height=0.0715, pads=0.035)
    assert result["success"], result["reason"]
    box = _box(mcp)
    box_top = box["position"][2] + box["dimensions"][2] / 2
    fingertips = _drop_heights(mcp)[-1] - 0.225
    assert fingertips - box_top == pytest.approx(0.01)
