from __future__ import annotations

import sys # :updated
import json
import subprocess
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
# updated: VENV_PYTHON = BASE_DIR / ".venv" / "bin" / "python"
VENV_PYTHON = Path(sys.executable)
FACTORY_PATH = BASE_DIR / "factory_twin.py"
RESULT_PATH = BASE_DIR / "simulation_result.csv"
STATUS_PATH = BASE_DIR / "agent_status.json"
EXECUTOR = ThreadPoolExecutor(max_workers=1)


# The local agent fills these values from the sidebar controls. Keeping the
# template in the app makes the project self-contained and avoids any API key.
FACTORY_TEMPLATE = '''#!/usr/bin/env python3
import csv
import random
from pathlib import Path

import simpy


ARRIVAL_INTERVAL = __ARRIVAL_INTERVAL__
PROCESSING_MEANS = [__STAGE1_MEAN__, __STAGE2_MEAN__, __STAGE3_MEAN__]
SIMULATION_RUNTIME = __SIMULATION_RUNTIME__
OUTPUT_PATH = Path(__file__).resolve().parent / "simulation_result.csv"
STAGE_NAMES = ["Stage 1", "Stage 2", "Stage 3"]
FIELDNAMES = [
    "part_id",
    "stage",
    "wait_time",
    "service_time",
    "lead_time",
    "completed_at",
]


def record_stage(writer, output_file, part_id, stage_name, wait_time, service_time, lead_time, completed_at):
    writer.writerow(
        {
            "part_id": part_id,
            "stage": stage_name,
            "wait_time": round(wait_time, 6),
            "service_time": round(service_time, 6),
            "lead_time": round(lead_time, 6),
            "completed_at": round(completed_at, 6),
        }
    )
    output_file.flush()


def process_part(env, part_id, stages, arrival_time, writer, output_file):
    for stage_name, resource, mean_service_time in stages:
        requested_at = env.now
        with resource.request() as request:
            yield request
            wait_time = env.now - requested_at
            service_time = random.expovariate(1.0 / mean_service_time) if mean_service_time > 0 else 0.0
            yield env.timeout(service_time)

        record_stage(
            writer=writer,
            output_file=output_file,
            part_id=part_id,
            stage_name=stage_name,
            wait_time=wait_time,
            service_time=service_time,
            lead_time=env.now - arrival_time,
            completed_at=env.now,
        )


def generate_parts(env, stages, writer, output_file):
    part_id = 1
    while env.now < SIMULATION_RUNTIME:
        env.process(
            process_part(
                env=env,
                part_id=part_id,
                stages=stages,
                arrival_time=env.now,
                writer=writer,
                output_file=output_file,
            )
        )
        part_id += 1
        yield env.timeout(ARRIVAL_INTERVAL)


def main():
    if ARRIVAL_INTERVAL <= 0:
        raise ValueError("ARRIVAL_INTERVAL must be greater than zero")
    if SIMULATION_RUNTIME <= 0:
        raise ValueError("SIMULATION_RUNTIME must be greater than zero")
    if any(mean <= 0 for mean in PROCESSING_MEANS):
        raise ValueError("Every processing mean must be greater than zero")

    random.seed(2026)
    environment = simpy.Environment()
    resources = [simpy.Resource(environment, capacity=1) for _ in STAGE_NAMES]
    stages = list(zip(STAGE_NAMES, resources, PROCESSING_MEANS))

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        output_file.flush()
        environment.process(generate_parts(environment, stages, writer, output_file))
        environment.run(until=SIMULATION_RUNTIME)


if __name__ == "__main__":
    main()
'''


def _write_status(state: str, trial: int, message: str, **extra: Any) -> None:
    payload = {
        "state": state,
        "trial": trial,
        "message": message,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    payload.update(extra)
    temporary_path = STATUS_PATH.with_name(".agent_status.tmp")
    try:
        temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary_path.replace(STATUS_PATH)
    except OSError:
        pass


def _render_factory_script(params: dict[str, float]) -> str:
    replacements = {
        "__ARRIVAL_INTERVAL__": repr(float(params["arrival_interval"])),
        "__STAGE1_MEAN__": repr(float(params["stage_1_mean"])),
        "__STAGE2_MEAN__": repr(float(params["stage_2_mean"])),
        "__STAGE3_MEAN__": repr(float(params["stage_3_mean"])),
        "__SIMULATION_RUNTIME__": repr(float(params["simulation_runtime"])),
    }
    script = FACTORY_TEMPLATE
    for placeholder, value in replacements.items():
        script = script.replace(placeholder, value)
    return script


def _local_repair_agent(
    params: dict[str, float], previous_script: str, traceback_text: str
) -> str:
    """Apply deterministic repairs after a failed local factory run.

    This is the offline replacement for an API-backed repair agent. The
    complete template is always available as a safe fallback, while the
    targeted replacements make common generated-script failures recoverable.
    """
    repaired = previous_script

    if "mean_service_time > 0" not in repaired:
        repaired = repaired.replace(
            "random.expovariate(1.0 / mean_service_time)",
            "random.expovariate(1.0 / mean_service_time) if mean_service_time > 0 else 0.0",
        )

    if "OUTPUT_PATH = Path(__file__).resolve().parent / \"simulation_result.csv\"" not in repaired:
        repaired = _render_factory_script(params)

    try:
        compile(repaired, str(FACTORY_PATH), "exec")
    except SyntaxError:
        repaired = _render_factory_script(params)

    # If a runtime traceback indicates a missing or damaged core section,
    # regenerate the known-good local program for the next trial.
    required_fragments = ("import simpy", "def process_part", "environment.run")
    if any(fragment not in repaired for fragment in required_fragments):
        repaired = _render_factory_script(params)

    return repaired


def _run_local_agent(params: dict[str, float]) -> dict[str, Any]:
    previous_script: str | None = None
    previous_traceback = ""
    max_trials = 3
    timeout_seconds = max(120, min(600, int(params["simulation_runtime"] * 2 + 60)))

    for trial in range(1, max_trials + 1):
        if previous_script is None:
            _write_status("running", trial, "Generating factory_twin.py from the local template")
            script = _render_factory_script(params)
        else:
            _write_status(
                "repairing",
                trial,
                "Inspecting the traceback and applying an offline self-healing repair",
                traceback=previous_traceback[-6000:],
            )
            script = _local_repair_agent(params, previous_script, previous_traceback)

        try:
            compile(script, str(FACTORY_PATH), "exec")
            FACTORY_PATH.write_text(script, encoding="utf-8")
        except Exception:
            previous_script = script
            previous_traceback = traceback.format_exc()
            _write_status(
                "retrying",
                trial,
                "The generated script did not compile; retrying the local repair loop",
                traceback=previous_traceback[-6000:],
            )
            continue

        try:
            RESULT_PATH.unlink()
        except FileNotFoundError:
            pass

        try:
            completed = subprocess.run(
                [str(VENV_PYTHON), str(FACTORY_PATH)],
                cwd=BASE_DIR,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            command_output = "\n".join(
                piece for piece in (completed.stdout, completed.stderr) if piece
            ).strip()
        except subprocess.TimeoutExpired as exc:
            command_output = f"factory_twin.py timed out after {timeout_seconds} seconds: {exc}"
            completed = None
        except Exception:
            command_output = traceback.format_exc()
            completed = None

        if completed is not None and completed.returncode == 0 and RESULT_PATH.exists():
            _write_status(
                "success",
                trial,
                "Simulation completed successfully",
                output=command_output[-6000:],
            )
            return {
                "success": True,
                "trial": trial,
                "output": command_output,
            }

        previous_script = script
        previous_traceback = command_output or "factory_twin.py exited without creating simulation_result.csv"
        if completed is not None and completed.returncode != 0:
            previous_traceback = (
                f"factory_twin.py exited with status {completed.returncode}\n{previous_traceback}"
            )
        _write_status(
            "retrying" if trial < max_trials else "failed",
            trial,
            "The simulation returned a traceback; the local self-healing loop is continuing",
            traceback=previous_traceback[-6000:],
        )

    _write_status(
        "failed",
        max_trials,
        "The local self-healing loop reached its three-trial limit",
        traceback=previous_traceback[-6000:],
    )
    return {
        "success": False,
        "trial": max_trials,
        "error": previous_traceback,
    }


def _read_status() -> dict[str, Any]:
    try:
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"state": "idle", "message": "Ready"}


def _load_results() -> pd.DataFrame | None:
    if not RESULT_PATH.exists():
        return None
    try:
        frame = pd.read_csv(RESULT_PATH)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return None
    if frame.empty:
        return frame
    for column in ("part_id", "wait_time", "service_time", "lead_time", "completed_at"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["part_id", "stage"])


def _render_results() -> None:
    frame = _load_results()
    if frame is None:
        st.info("Run the local agent to generate simulation_result.csv and display the results.")
        return
    if frame.empty:
        st.warning("The simulation produced an empty result file. Try a longer runtime.")
        return

    final_stage = frame[frame["stage"] == "Stage 3"]
    throughput = int(final_stage["part_id"].nunique()) if not final_stage.empty else 0
    average_lead_time = (
        float(final_stage["lead_time"].mean()) if not final_stage.empty else 0.0
    )
    average_wait_time = float(frame["wait_time"].mean()) if "wait_time" in frame else 0.0

    metric_one, metric_two, metric_three = st.columns(3)
    metric_one.metric("Total throughput", f"{throughput:,} parts")
    metric_two.metric("Average lead time", f"{average_lead_time:.2f} time units")
    metric_three.metric("Average wait time", f"{average_wait_time:.2f} time units")

    st.subheader("Wait times by process stage")
    wait_frame = frame[["part_id", "stage", "wait_time"]].dropna().sort_values("part_id")
    if not wait_frame.empty:
        wait_chart = px.line(
            wait_frame,
            x="part_id",
            y="wait_time",
            color="stage",
            markers=True,
            labels={
                "part_id": "Part",
                "wait_time": "Wait time",
                "stage": "Process stage",
            },
            title="Queue wait time for each completed stage",
        )
        st.plotly_chart(wait_chart, use_container_width=True)

    if not final_stage.empty:
        lead_chart = px.line(
            final_stage.sort_values("part_id"),
            x="part_id",
            y="lead_time",
            markers=True,
            labels={"part_id": "Part", "lead_time": "Lead time"},
            title="Lead time for parts completed by Stage 3",
        )
        st.plotly_chart(lead_chart, use_container_width=True)

    with st.expander("View simulation_result.csv"):
        st.dataframe(frame, use_container_width=True, hide_index=True)


st.set_page_config(page_title="Offline Digital Twin", page_icon="🏭", layout="wide")

if "agent_future" not in st.session_state:
    st.session_state.agent_future = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "run_params" not in st.session_state:
    st.session_state.run_params = None


st.title("Offline Digital Twin Simulation")
st.caption(
    "A local, API-free simulation agent generates factory_twin.py, executes it in .venv, "
    "and retries traceback repairs up to three times."
)

with st.sidebar:
    st.header("Simulation parameters")
    arrival_interval = st.slider(
        "Part arrival interval",
        min_value=0.1,
        max_value=20.0,
        value=1.0,
        step=0.1,
        help="Mean time between incoming parts.",
    )
    stage_1_mean = st.slider(
        "Stage 1 processing mean",
        min_value=0.1,
        max_value=20.0,
        value=1.5,
        step=0.1,
    )
    stage_2_mean = st.slider(
        "Stage 2 processing mean",
        min_value=0.1,
        max_value=20.0,
        value=2.0,
        step=0.1,
    )
    stage_3_mean = st.slider(
        "Stage 3 processing mean",
        min_value=0.1,
        max_value=20.0,
        value=1.2,
        step=0.1,
    )
    simulation_runtime = st.slider(
        "Total simulation runtime",
        min_value=10.0,
        max_value=600.0,
        value=60.0,
        step=5.0,
        help="Simulation time units, not wall-clock seconds.",
    )

    current_future = st.session_state.agent_future
    currently_running = current_future is not None and not current_future.done()
    run_clicked = st.button(
        "Run / regenerate twin",
        type="primary",
        use_container_width=True,
        disabled=currently_running,
    )
    if run_clicked:
        parameters = {
            "arrival_interval": float(arrival_interval),
            "stage_1_mean": float(stage_1_mean),
            "stage_2_mean": float(stage_2_mean),
            "stage_3_mean": float(stage_3_mean),
            "simulation_runtime": float(simulation_runtime),
        }
        st.session_state.run_params = parameters
        st.session_state.last_result = None
        st.session_state.agent_future = EXECUTOR.submit(_run_local_agent, parameters)
        st.rerun()

    st.divider()
    st.caption("No API key, model subscription, or external agent is required.")


future = st.session_state.agent_future
if future is not None and not future.done():
    status = _read_status()
    st.info(f"{status.get('message', 'Local agent is working')} — trial {status.get('trial', 1)} of 3")
    _render_results()
    time.sleep(0.8)
    st.rerun()

if future is not None and future.done():
    try:
        st.session_state.last_result = future.result()
    except Exception:
        st.session_state.last_result = {
            "success": False,
            "error": traceback.format_exc(),
        }
    st.session_state.agent_future = None


result = st.session_state.last_result
if result is not None:
    if result.get("success"):
        st.success(f"Simulation completed on local trial {result.get('trial', 1)}.")
    else:
        st.error("The local agent could not complete the simulation after three trials.")
        st.code(result.get("error", "Unknown simulation error"), language="text")

_render_results()
