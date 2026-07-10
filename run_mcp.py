#!/usr/bin/env python3
"""Remote LabOS MCP runner for the robot protocol runtime."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from labos.mcp import Context, RemoteMCP
from labos.mcp.protocol import ToolEvent, ToolFatal, ToolHeartbeat

LOGGER = logging.getLogger("run_mcp")
LOCK_MESSAGE = "Error: MCP server locked due to a prior fatal robot error. Please restart this script."

_mcp_locked = False
_progress_watcher_task: asyncio.Task[None] | None = None


class LoggingRemoteMCP(RemoteMCP):
    """RemoteMCP subclass that logs heartbeat activity for local operators."""

    async def _heartbeat_loop(self, ws: Any) -> None:
        while True:
            await _send_ws_payload(ws, ToolHeartbeat().model_dump(mode="json"))
            LOGGER.info("Heartbeat sent to LabOS relay")
            await asyncio.sleep(self.heartbeat_interval)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _mock_enabled() -> bool:
    return _env_bool("MOCK_MODE", False)


def _mask_secret_tail(value: str, keep: int = 4) -> str:
    """Mask a secret while keeping only the last few characters visible."""
    if not value:
        return "<unset>"
    if keep <= 0:
        return "*" * len(value)
    visible = value[-keep:] if len(value) > keep else value
    masked_len = max(0, len(value) - len(visible))
    return ("*" * masked_len) + visible


def _load_dotenv_into_environ(path: Path | None = None) -> dict[str, str]:
    """Parse .env (matching the labos SDK loader) and set values in os.environ.

    Existing environment variables take precedence so explicit shell exports
    always win. Returns the parsed mapping for diagnostics.
    """
    dotenv_path = Path(path) if path is not None else (ROOT / ".env")
    values: dict[str, str] = {}
    if not dotenv_path.is_file():
        return values
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].lstrip()
        key, separator, raw_value = stripped.partition("=")
        if not separator:
            continue
        key = key.strip()
        if not key:
            continue
        value = raw_value.strip()
        if (value.startswith("\"") and value.endswith("\"")) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        values[key] = value
        os.environ.setdefault(key, value)
    return values


def _require_labos_settings() -> None:
    """Exit with a clear message if required LabOS settings are missing."""
    missing: list[str] = []
    if not os.environ.get("LABOS_URL"):
        missing.append("LABOS_URL")
    if not os.environ.get("LABOS_API_KEY"):
        missing.append("LABOS_API_KEY")
    if missing:
        joined = ", ".join(missing)
        LOGGER.error(
            "Missing required LabOS settings: %s. Set them in .env, the environment, or via constructor arguments.",
            joined,
        )
        raise SystemExit(2)


def configure_logging(verbose: bool = False, *, mock: bool = False) -> None:
    """Configure console + file logging under logs/."""
    logs_dir = ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / ("run_mock_mcp.log" if mock else "run_mcp.log")
    level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    LOGGER.info("Logging to %s", log_file)


def _progress_value(status: dict[str, Any]) -> float:
    pct = float(status.get("progress_pct") or 0.0)
    return max(0.0, min(1.0, pct / 100.0))


def _status_message(status: dict[str, Any]) -> str:
    description = status.get("current_step_description")
    if description:
        return str(description)
    index = int(status.get("current_step_index") or 0)
    total = int(status.get("total_steps") or 0)
    if total:
        return f"step {index + 1}/{total}"
    return str(status.get("state") or "idle")


def _ensure_vision_display(headless: bool) -> None:
    if headless or _mock_enabled():
        LOGGER.info("Vision display disabled (HEADLESS=%s, MOCK_MODE=%s)", headless, _mock_enabled())
        return
    try:
        from aira.robot import start_vision_display

        start_vision_display()
        LOGGER.info("Vision display started")
    except BaseException:
        LOGGER.warning("Vision display failed to start; MCP server will continue", exc_info=True)


def _lock_mcp(error_message: str) -> None:
    global _mcp_locked
    _mcp_locked = True
    LOGGER.error("MCP locked due to fatal robot error. Please restart this script to clear. Error: %s", error_message)


def _disable_robot_best_effort() -> None:
    if _mock_enabled():
        return
    try:
        from aira.robot import arm

        a = arm()
        try:
            a.clear_error()
        except Exception:
            pass
        a.set_position_mode()
    except Exception:
        LOGGER.warning("Failed to disable robot after fatal error", exc_info=True)


async def _send_ws_payload(ws: Any, payload: dict[str, Any]) -> None:
    await ws.send(json.dumps(payload))


async def _send_background_event(ws: Any, name: str, payload: dict[str, Any]) -> None:
    event = ToolEvent(name=name, payload=payload)
    await _send_ws_payload(ws, event.model_dump(mode="json", exclude_none=True))


async def _send_background_fatal(ws: Any, code: str, message: str, **fields: Any) -> None:
    fatal = ToolFatal(code=code, message=message, fields=fields)
    await _send_ws_payload(ws, fatal.model_dump(mode="json", exclude_none=True))


async def _handle_failed_status(
    status: dict[str, Any],
    *,
    ctx: Context | None = None,
    ws: Any | None = None,
) -> None:
    error = str(status.get("error") or "Unknown protocol failure")
    if _mcp_locked:
        return
    _disable_robot_best_effort()
    _lock_mcp(error)
    fields = {
        "protocol": status.get("protocol_name"),
        "step": status.get("current_step_index"),
        "step_name": status.get("current_step_name"),
    }
    if ctx is not None:
        await ctx.fatal("xarm_error", error, **fields)
    if ws is not None:
        await _send_background_fatal(ws, "xarm_error", error, **fields)


async def _watch_protocol_status(ws: Any, interval: float = 0.35) -> None:
    from aira import protocol_runner

    last_snapshot: tuple[Any, ...] | None = None
    while True:
        event = protocol_runner.get_status_changed_event()
        await asyncio.to_thread(event.wait, timeout=interval)
        event.clear()
        status = protocol_runner.get_protocol_status()
        snapshot = (
            status.get("state"),
            status.get("protocol_name"),
            status.get("current_step_index"),
            status.get("current_step_description"),
            status.get("progress_pct"),
            status.get("error"),
        )
        if snapshot == last_snapshot:
            continue
        last_snapshot = snapshot
        LOGGER.info(
            "Protocol status: state=%s protocol=%s step=%s/%s name=%s description=%s progress=%.1f%%",
            status.get("state"),
            status.get("protocol_name"),
            int(status.get("current_step_index") or 0) + 1,
            status.get("total_steps"),
            status.get("current_step_name"),
            _status_message(status),
            float(status.get("progress_pct") or 0.0),
        )
        await _send_background_event(ws, "protocol.progress", status)
        if status.get("state") == "failed":
            await _handle_failed_status(status, ws=ws)


def _locked() -> bool:
    return _mcp_locked


async def _mock_info(ctx: Context, message: str, **fields: Any) -> None:
    await ctx.info(f"[mock] {message}", **fields)


def create_mcp(*, headless: bool | None = None, mock: bool | None = None) -> RemoteMCP:
    if mock is not None:
        os.environ["MOCK_MODE"] = "1" if mock else "0"
    actual_headless = _env_bool("HEADLESS", False) if headless is None else headless
    heartbeat = float(os.environ.get("LABOS_HEARTBEAT_INTERVAL", "15"))
    mcp = LoggingRemoteMCP(
        url=os.environ.get("LABOS_URL", ""),
        api_key=os.environ.get("LABOS_API_KEY", ""),
        device_id=os.environ.get("LABOS_DEVICE_ID", "labos-robot-runtime"),
        name="labos-robot-runtime",
        heartbeat_interval=heartbeat,
        metadata={
            "runtime": "LabOS-Robot-Runtime",
            "mock_mode": _mock_enabled(),
            "headless": actual_headless,
        },
    )

    @mcp.on_startup
    async def _startup() -> None:
        LOGGER.info("MCP startup (mock=%s, headless=%s)", _mock_enabled(), actual_headless)
        _ensure_vision_display(actual_headless)

    @mcp.on_connect
    async def _connected(ws: Any) -> None:
        global _progress_watcher_task
        LOGGER.info("Connected to LabOS relay (mock=%s)", _mock_enabled())
        if _progress_watcher_task is not None:
            _progress_watcher_task.cancel()
            await asyncio.gather(_progress_watcher_task, return_exceptions=True)
        _progress_watcher_task = asyncio.create_task(_watch_protocol_status(ws))

    @mcp.on_disconnect
    async def _disconnected(_ws: Any) -> None:
        global _progress_watcher_task
        LOGGER.info("Disconnected from LabOS relay (mock=%s)", _mock_enabled())
        if _progress_watcher_task is not None:
            _progress_watcher_task.cancel()
            await asyncio.gather(_progress_watcher_task, return_exceptions=True)
            _progress_watcher_task = None

    @mcp.tool()
    async def get_status(ctx: Context) -> str:
        """Get the current robot protocol status."""
        if _locked():
            return LOCK_MESSAGE
        from aira.protocol_runner import get_protocol_status_formatted

        return get_protocol_status_formatted()

    @mcp.tool()
    async def get_protocols(ctx: Context) -> str:
        """List available YAML protocols with brief descriptions and major steps."""
        if _locked():
            return LOCK_MESSAGE
        from aira.protocol_runner import list_protocols

        protocols = list_protocols()
        if not protocols:
            return "No protocols found in protocols/."
        lines: list[str] = []
        for protocol in protocols:
            lines.append(f"- {protocol.get('name', '')}: {protocol.get('brief', '')}")
            for step in protocol.get("steps") or []:
                lines.append(f"  - {step}")
        return "\n".join(lines)

    @mcp.tool()
    async def describe_protocol(protocol_name: str, ctx: Context) -> str:
        """Describe a protocol by name: brief and major steps."""
        if _locked():
            return LOCK_MESSAGE
        from aira.protocol_runner import describe_protocol as _describe

        name = (protocol_name or "").strip()
        if not name:
            return "Error: protocol_name is required."
        try:
            meta = _describe(name)
        except FileNotFoundError as exc:
            return f"Error: {exc}"
        out = [f"Protocol: {meta.get('name', name)}", f"Brief: {meta.get('brief', '')}", "Major steps:"]
        for step in meta.get("steps") or []:
            out.append(f"  - {step}")
        return "\n".join(out)

    @mcp.tool()
    async def get_object_definitions(ctx: Context) -> str:
        """List exact configured object names and their shapes from configs/objects.yaml."""
        if _locked():
            return LOCK_MESSAGE
        from aira.robot import get_object_definitions as _get_defs

        defs = _get_defs()
        if not defs:
            return "No object definitions in configs/objects.yaml."
        lines = []
        for obj in defs:
            lines.append(
                f"- \"{obj.get('name', '?')}\": shape {obj.get('shape_size_mm', '?')}, "
                f"yolo_class \"{obj.get('yolo_class', '')}\", pick_type {obj.get('pick_type', 'toolhead_close')}"
            )
        return "Configured objects (use the name in quotes exactly for move_to_object):\n" + "\n".join(lines)

    @mcp.tool()
    async def start_protocol(protocol_name: str, ctx: Context, blocking: bool = False) -> str:
        """Start a YAML protocol by name. Set blocking=true to stream progress until completion."""
        if _locked():
            return LOCK_MESSAGE
        from aira import protocol_runner

        name = (protocol_name or "").strip()
        if not name:
            return "Error: protocol_name is required (e.g. 'vortexing', 'test')."
        try:
            started = protocol_runner.run_protocol(name, mock=protocol_runner.is_mock_mode())
        except FileNotFoundError as exc:
            return f"Error: Protocol not found. {exc}. Protocols dir: {protocol_runner.PROTOCOLS_DIR}"
        if not started:
            return "A protocol is already running. Use get_status to see current state."

        LOGGER.info("Started protocol '%s' (blocking=%s, mock=%s)", name, blocking, protocol_runner.is_mock_mode())
        await ctx.emit("protocol.started", {"protocol": name, "mock_mode": protocol_runner.is_mock_mode()})
        if not blocking:
            return f"Started protocol '{name}'."

        await ctx.progress(0.0, f"Started protocol '{name}'.")
        while True:
            if ctx.cancelled:
                protocol_runner.stop_protocol()
                return f"Cancelled protocol '{name}'."
            event = protocol_runner.get_status_changed_event()
            await asyncio.to_thread(event.wait, timeout=0.35)
            event.clear()
            status = protocol_runner.get_protocol_status()
            LOGGER.info(
                "Running protocol '%s': step %s/%s %s - %s",
                status.get("protocol_name"),
                int(status.get("current_step_index") or 0) + 1,
                status.get("total_steps"),
                status.get("current_step_name"),
                _status_message(status),
            )
            await ctx.progress(_progress_value(status), _status_message(status))
            await ctx.info("protocol status", **status)
            if status.get("state") == "failed":
                await _handle_failed_status(status, ctx=ctx)
            if status.get("state") == "finished":
                return f"Protocol '{name}' finished successfully."

    @mcp.tool()
    async def stop_robot(ctx: Context) -> str:
        """Stop the current protocol and return the robot to position control mode."""
        if _locked():
            return LOCK_MESSAGE
        from aira.protocol_runner import stop_protocol

        stop_protocol()
        if _mock_enabled():
            await _mock_info(ctx, "stop_robot")
            return "[mock] success: protocol stopped"
        try:
            from aira.robot import arm

            arm().set_position_mode()
        except Exception as exc:
            return f"Stopped protocol, but failed to set position mode: {exc}"
        return "Stopped. Protocol stopped and robot in position control mode."

    @mcp.tool()
    async def manual_mode(ctx: Context) -> str:
        """Put the default robot arm in manual teaching mode."""
        if _locked():
            return LOCK_MESSAGE
        if _mock_enabled():
            await _mock_info(ctx, "manual_mode")
            return "[mock] success: robot is now in manual mode"
        try:
            from aira.robot import arm

            arm().set_manual_mode()
            return "Robot is now in manual mode. Use stop_robot to return to position control."
        except Exception as exc:
            return f"Error: {exc}"

    @mcp.tool()
    async def go_home(ctx: Context) -> str:
        """Send the default robot arm to the recorded home location."""
        if _locked():
            return LOCK_MESSAGE
        if _mock_enabled():
            await _mock_info(ctx, "go_home")
            return "[mock] success: robot moved to home"
        try:
            from aira.robot import arm

            a = arm()
            a.set_position_mode()
            code = a.go_to("home")
            if code == 0:
                return "Robot moved to home."
            if a.check_error():
                a.clear_error()
            return f"go_to returned code {code}."
        except Exception as exc:
            return f"Error: {exc}"

    @mcp.tool()
    async def gripper(position: str, ctx: Context) -> str:
        """Control the gripper: close, midway, open, or a numeric 0-800 position."""
        if _locked():
            return LOCK_MESSAGE
        positions = {"close": 0, "closed": 0, "midway": 300, "half": 300, "open": 600}
        pos_str = (position or "").strip().lower()
        if pos_str in positions:
            pos = positions[pos_str]
        else:
            try:
                pos = int(float(pos_str))
            except ValueError:
                return f"Invalid position '{position}'. Use 'close', 'midway', 'open', or a number 0-800."
        pos = max(0, min(800, pos))
        if _mock_enabled():
            await _mock_info(ctx, "gripper", position=pos)
            return f"[mock] success: gripper moved to position {pos}"
        try:
            from aira.robot import arm

            code = arm().set_gripper_position(pos, wait=True)
            return f"Gripper moved to position {pos}." if code == 0 else f"Gripper command returned code {code}."
        except Exception as exc:
            return f"Error: {exc}"

    @mcp.tool()
    async def z_level(level: str, ctx: Context) -> str:
        """Move the robot to a predefined or numeric Z height in millimeters."""
        if _locked():
            return LOCK_MESSAGE
        levels = {"low": 115, "medium": 200, "med": 200, "high": 300}
        level_str = (level or "").strip().lower()
        if level_str in levels:
            height = levels[level_str]
        else:
            try:
                height = float(level_str)
            except ValueError:
                return f"Invalid level '{level}'. Use 'low', 'medium', 'high', or a number in mm."
        if _mock_enabled():
            await _mock_info(ctx, "z_level", height=height)
            return f"[mock] success: robot moved to z_level {height}mm"
        try:
            from aira.robot import arm

            a = arm()
            a.set_position_mode()
            code = a.z_level(height, wait=True)
            if code == 0:
                return f"Robot moved to z_level {height}mm."
            if a.check_error():
                a.clear_error()
            return f"z_level command returned code {code}."
        except Exception as exc:
            return f"Error: {exc}"

    @mcp.tool()
    async def is_holding_something(ctx: Context) -> Any:
        """Return whether the gripper position suggests the robot is holding something."""
        if _locked():
            return LOCK_MESSAGE
        if _mock_enabled():
            await _mock_info(ctx, "is_holding_something")
            return False
        try:
            from aira.robot import arm

            code, pos = arm().get_gripper_position()
            return bool(code == 0 and float(pos) < 300)
        except Exception:
            return False

    @mcp.tool()
    async def list_objects(ctx: Context) -> str:
        """List objects visible in the current camera frame."""
        if _locked():
            return LOCK_MESSAGE
        if _mock_enabled():
            await _mock_info(ctx, "list_objects")
            return "[mock] success: 50ml eppendorf at (320, 240) - orange, 450mm depth"
        from aira.robot import get_latest_detections_detailed

        detections = get_latest_detections_detailed()
        if not detections:
            return "No objects detected (vision may not be running or no objects in view)."
        parts = []
        for det in detections:
            name = det.get("object_name", "unknown")
            cx, cy = det.get("center_px", (0, 0))
            color = det.get("color", "unknown")
            depth_mm = det.get("depth_mm")
            if depth_mm is not None:
                parts.append(f"{name} at ({cx}, {cy}) - {color}, {depth_mm}mm depth")
            else:
                parts.append(f"{name} at ({cx}, {cy}) - {color}")
        return "; ".join(parts)

    @mcp.tool()
    async def see_object(object_name: str, ctx: Context) -> Any:
        """Return whether a configured object is visible."""
        if _locked():
            return LOCK_MESSAGE
        if _mock_enabled():
            await _mock_info(ctx, "see_object", object=object_name)
            return True
        from aira.robot import see_object as _see_object

        return _see_object(object_name or "")

    @mcp.tool()
    async def move_to_object(
        object_name: str,
        ctx: Context,
        target_px_x: int | None = None,
        target_px_y: int | None = None,
    ) -> str:
        """Move robot to an object visible in the camera frame."""
        if _locked():
            return LOCK_MESSAGE
        if _mock_enabled():
            await _mock_info(ctx, "move_to_object", object=object_name, target_px_x=target_px_x, target_px_y=target_px_y)
            return f"[mock] success: moved to '{object_name}'"
        try:
            from aira.robot import arm

            a = arm()
            a.set_position_mode()
            pick_type: Any = "toolhead_close"
            if target_px_x is not None and target_px_y is not None:
                pick_type = (float(target_px_x), float(target_px_y))
            result = a.move_to_object(object_name, pick_type=pick_type, display=False)
            if result.get("success"):
                return f"Moved to '{object_name}'. {result.get('moves_done', 0)} correction move(s) made."
            return f"Failed to move to '{object_name}': {result.get('error', 'unknown error')}"
        except Exception as exc:
            return f"Error: {exc}"

    return mcp


def main() -> int:
    parser = argparse.ArgumentParser(description="LabOS Robot Runtime remote MCP server.")
    parser.add_argument("--headless", action="store_true", help="Disable the camera vision display window.")
    parser.add_argument("--mock", action="store_true", help="Run in MOCK_MODE without robot or camera hardware.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    if args.mock:
        os.environ["MOCK_MODE"] = "1"
    if args.headless:
        os.environ["HEADLESS"] = "1"
    _load_dotenv_into_environ()
    configure_logging(args.verbose, mock=args.mock or _mock_enabled())
    LOGGER.info(
        "Remote MCP config: url=%s device_id=%s api_key=%s",
        os.environ.get("LABOS_URL", "<unset>"),
        os.environ.get("LABOS_DEVICE_ID", "<auto>"),
        _mask_secret_tail(os.environ.get("LABOS_API_KEY", ""), keep=4),
    )
    _require_labos_settings()

    try:
        LOGGER.info("Starting LabOS Robot Runtime MCP server")
        create_mcp(headless=args.headless or _env_bool("HEADLESS", False), mock=args.mock or _mock_enabled()).run()
        return 0
    except KeyboardInterrupt:
        return 0
    except BaseException:
        LOGGER.exception("MCP server exited with error")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
