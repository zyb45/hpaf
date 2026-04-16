# HPAF: Hierarchical Program-Aided Framework for Real-World Robotic Task Execution

> A practical stage-one implementation of our HPAF idea: a complex language instruction is decomposed into atomic tasks, each atomic task is compiled into an executable program over a bounded runtime API, executed on a real PiPER arm with RGB-D perception, and confirmed before the pipeline proceeds to the next step.

## Overview

This repository is the first runnable real-world prototype of our HPAF pipeline. It targets **complex tabletop manipulation tasks** and asks a concrete question:

**Can an LLM decompose a complex user instruction, generate executable programs grounded in a small robot API library, and complete a real hardware task loop?**

For the stage-one minimal demo in this repository, the answer is **yes**.

The current implementation connects:

- a **TaskAgent** for complex-task decomposition,
- a **ProgramAgent** for API-grounded executable code generation,
- a **foundation-vision-based perception stack**,
- a **PiPER robot backend**,
- and a **task-verification module** that supports both **manual confirmation** and **AI-based atomic-task verification**.

The project follows the core idea below:

1. A user provides a **complex task**.
2. HPAF performs the **first-layer decomposition** and splits it into a sequence of **atomic tasks**.
3. Each atomic task is then handled by a **second-layer execution loop**:
   - **Perception**: detect the relevant object or target region from the current image.
   - **Execution**: generate and run an executable program over the runtime API.
   - **Confirmation**: decide whether the current atomic task has been completed.
4. Once the current atomic task is confirmed complete, the system moves on to the next atomic task until the whole instruction is finished.

This repository is therefore an engineering realization of the main paper direction: **dual-layer decomposition + automatic program generation from bottom-level APIs**.

---

## What This Repository Contributes

### 1. Dual-layer decomposition for embodied execution

Instead of treating the problem as a flat “instruction -> action” mapping, HPAF explicitly introduces two layers:

- **Layer 1: complex task -> atomic tasks**
- **Layer 2: atomic task -> perception / execution / confirmation**

This improves controllability, interpretability, and debugging efficiency in real-world robotic execution.

### 2. Program generation over a bounded runtime API

Rather than asking the model to output unconstrained free-form plans, HPAF exposes a **runtime API library** and requires the ProgramAgent to generate a **directly executable Python script body** over that API.

As a result, generated code is:

- grounded in available robot capabilities,
- easier to inspect,
- easier to debug,
- and much closer to a deployable robotics workflow than pure natural-language planning.

### 3. Unified atomic-task verification module

Beyond code generation and execution, the repository now includes a **unified verification module** for atomic-task completion.

The key design is that generated programs do **not** rely on object-specific hard-coded verification logic at the script level. Instead, they terminate with a **generic verification call**:

```python
result = ai_verify_atomic_task()
```

This function uses:

- the **current atomic task text**,
- the **post-execution observation image**,
- and the **global auxiliary camera view**,

then asks the LLM to judge whether the current atomic task has been successfully completed.

This design improves:

- **generality**, because the same verification entry point is reused across different atomic tasks,
- **cleanliness of generated programs**, because ad hoc task-specific verification logic is avoided,
- and **pipeline consistency**, because both `manual` and `auto` modes can share the same completion signal.

### 4. Human-confirmed stage-one real-world closed loop

This repository focuses on a practical first milestone: **running the full head-to-tail loop on a real robot**.

The system already completes:

- task decomposition,
- program generation,
- real execution through a robot backend,
- and per-atomic-task completion confirmation.

At this stage, HPAF supports two confirmation styles:

- **manual confirmation**, where the user remains in the loop,
- **AI-based verification**, where the program itself outputs a boolean result via `ai_verify_atomic_task()`.

This makes the repository a meaningful transition point between a human-supervised stage-one system and a more autonomous embodied-agent pipeline.

### 5. Chinese prompting for better practical generation quality

The prompt templates in this project are intentionally written mainly in **Chinese**, while most task outputs and generated executable programs are required to be in **English**.

This is a deliberate design choice. In our experiments on the current platform, Chinese prompting produced **more stable and more accurate decomposition / code-generation behavior**. We therefore keep Chinese prompts in the released version and document this choice explicitly rather than forcing an English-only prompting setup.

---

## System Pipeline

## 1. TaskAgent: first-layer decomposition

Input:

- a global-view image,
- a user instruction describing a complex task.

Output:

- a scene summary,
- a list of atomic tasks.

The TaskAgent is required to split the task into fine-grained steps such as:

- `Grab the blue rectangular prism`
- `Put the grabbed blue rectangular prism into the red metal box`

A grasp step and a placement step are never merged into a single atomic task.

## 2. ProgramAgent: second-layer executable generation

For each atomic task, the ProgramAgent receives:

- the current close-up image,
- the current atomic task,
- the execution mode,
- the runtime API documentation.

It then outputs:

- `plan_brief`
- `program`

The generated program is constrained to a **linear executable structure** and is only allowed to call registered runtime APIs.

In the current implementation, each generated atomic program is organized in the form of:

- **perception**
- **alignment**
- **interaction**
- **verification**

This is the executable realization of the broader second-layer design:

- **perception**
- **execution**
- **confirmation**

## 3. Execution

In `manual` mode, the system saves the generated script under the corresponding atomic-task log directory, and the user runs it in another terminal.

In `review` or `auto` mode, the system can directly execute the generated program through the executor.

## 4. Confirmation and Verification

After execution, the system checks whether the current atomic task has been completed.

There are now two supported paths:

### Manual path

In `manual` mode, the user can still inspect execution results and decide whether to continue to the next atomic task. At the same time, the generated program also ends with:

```python
result = ai_verify_atomic_task()
```

which provides an LLM-based completion judgment for the current atomic task.

### Automatic path

In `auto` mode, the pipeline executes each generated program directly and uses the returned boolean result from:

```python
result = ai_verify_atomic_task()
```

as the completion signal for deciding whether to proceed to the next atomic task.

This means that **manual and auto modes now share the same generic AI verification backend**, while differing only in whether the human remains in the execution loop.

---

## Runtime API Design

The ProgramAgent is not allowed to invent arbitrary functions. It must use the provided runtime API only. The current API includes:

- `debug`
- `detect_object_by_text`
- `estimate_top_grasp_pose`
- `build_pregrasp_pose`
- `estimate_place_pose`
- `open_gripper`
- `close_gripper`
- `stabilize_grasp`
- `move_to_pose`
- `retreat`
- `return_to_observe_pose`
- `ai_verify_atomic_task`

The verification stage is intentionally centered on the **generic** API:

- `ai_verify_atomic_task`

rather than task-specific verification calls embedded in generated programs.

This bounded API design is essential because it:

- narrows the code-generation action space,
- improves execution safety,
- makes generated programs much more interpretable,
- and keeps the verification logic unified across different atomic tasks.

---

## AI Verification Module

The newly added verification module is designed to judge whether an atomic task has succeeded **after the generated program finishes execution**.

### Verification entry point

Every generated atomic-task program is expected to end with:

```python
result = ai_verify_atomic_task()
```

### What `ai_verify_atomic_task()` uses

The function gathers:

- the **current atomic task description**,
- the **latest post-execution observation**,
- the **global-view auxiliary image** from the secondary camera,
- and the current execution context.

It then sends this information to the LLM and requests a structured success/failure judgment for the current atomic task.

### Why this matters

This design has three advantages:

1. **Task-level generality**  
   The same verification function can judge grasping tasks, placement tasks, and future atomic-task variants without requiring new task-specific verification APIs.

2. **Cleaner generated programs**  
   The ProgramAgent no longer needs to invent or call mid-program object-specific verification utilities such as `verify_object_grasped(...)`.

3. **Shared logic across modes**  
   Manual and automatic execution both depend on the same verification mechanism, making the system behavior more consistent and easier to debug.

### Verification artifacts

The verification result is saved to the corresponding atomic-task log directory, typically as a structured artifact such as:

```text
artifacts/ai_verify_output.json
```

This allows later inspection of:

- the verification prompt context,
- the model judgment,
- and the final success/failure decision.

---

## Hardware and Software Stack

### Hardware

- **Robot arm**: PiPER
- **Primary camera**: Gemini RGB-D camera mounted eye-in-hand
- **Secondary camera**: Astra camera used as a global auxiliary view
- **Depth source for manipulation**: the primary eye-in-hand RGB-D camera

### Software

- Python package project structure (`pyproject.toml`)
- ROS2-based camera bridge
- PiPER SDK backend
- Doubao API as the LLM backend
- Foundation-vision perception stack combining:
  - Florence-2
  - GroundingDINO
  - RGB-D geometry projection

### Coordinate convention

The project is built around the **eye-in-hand camera transform**. The current implementation uses the hand-eye calibration result configured in the repository and performs manipulation-oriented perception under that convention.

---

## Repository Structure

```text
hpaf/
├── configs/
│   ├── demo.yaml
│   ├── prompts.yaml
│   └── api_registry.yaml
├── hpaf/
│   ├── agents/
│   │   ├── task_agent.py
│   │   ├── program_agent.py
│   │   └── verify_agent.py
│   ├── api/
│   │   └── runtime_api.py
│   ├── camera/
│   │   └── shared_dir_camera.py
│   ├── core/
│   │   ├── app.py
│   │   ├── config.py
│   │   └── models.py
│   ├── execution/
│   │   ├── executor.py
│   │   └── program_validator.py
│   ├── geometry/
│   │   └── transforms.py
│   ├── llm/
│   │   ├── factory.py
│   │   └── openai_compatible.py
│   ├── perception/
│   │   ├── foundation_vision_perception.py
│   │   ├── llm_perception.py
│   │   └── classic_cv_perception.py
│   ├── pipeline/
│   │   └── orchestrator.py
│   └── robot/
│       ├── piper_backend.py
│       └── dummy_backend.py
├── ros_bridge/
│   ├── ros2_camera_dump.py
│   └── ros2_dual_camera_dump.py
├── scripts/
│   ├── run_pipeline.py
│   └── ...
├── tools/
│   └── init_stage1_env.sh
└── shared_scene/
```

---

## Installation

### 1. Enter the project and activate the environment

```bash
cd hpaf
conda activate piper
```

### 2. Install the package

```bash
python -m pip install -e .
```

### 3. Set the API key

```bash
export ARK_API_KEY=YOUR_API_KEY
```

### 4. Recommended proxy cleanup

```bash
unset ALL_PROXY
unset all_proxy
```

---

## Environment Startup

A convenient stage-one startup script is provided:

```bash
bash tools/init_stage1_env.sh
```

This script is used to bring up the first-stage real-execution environment, including:

- Gemini camera
- Gemini mirror/depth-related service handling
- Astra camera
- ROS bridge for dumping camera data into shared directories
- CAN activation
- gamepad / human control terminal
- RViz (optional)
- project terminal

The project relies on the shared camera directories:

- `shared_scene/primary`
- `shared_scene/secondary`

---

## Configuration

The main configuration file is:

```text
configs/demo.yaml
```

Important fields include:

- LLM backend (`doubao-seed-2-0-pro-260215`)
- primary / secondary shared camera directories
- primary extrinsic (`./assets/eyeinhand.json`)
- observe pose
- gripper width and force
- perception backend (`foundation_vision`)
- Florence-2 and GroundingDINO paths
- pipeline retry settings
- manual / review / auto execution mode behavior
- AI verification behavior and log output

---

## Quick Start

Run the pipeline with a complex task:

```bash
python scripts/run_pipeline.py \
  --config configs/demo.yaml \
  --task "Put the blue rectangular prism into the red metal box,Put the green cube into the yellow box." \
  --mode manual
```

Available modes:

- `manual`: generate each atomic-task script, save it, run it manually in another terminal, then inspect / confirm
- `review`: generate and display the program, ask for approval before direct execution
- `auto`: direct execution and automatic continuation based on `ai_verify_atomic_task()`

For `manual` mode, inspect the generated program and verify that it ends with:

```python
result = ai_verify_atomic_task()
```

---

## Minimal Real-World Demo

### Task

```text
Put the blue rectangular prism into the red metal box,Put the green cube into the yellow box.
```

### First-layer decomposition result

The TaskAgent decomposed the complex instruction into four atomic tasks:

1. `Grab the blue rectangular prism`
2. `Put the grabbed blue rectangular prism into the red metal box`
3. `Grab the green cube`
4. `Put the grabbed green cube into the yellow box`

### Example generated program for Atomic Task 1

For the first atomic task, the ProgramAgent generated a grasp program following this sequence:

- detect the target object,
- estimate the top grasp pose,
- build a pre-grasp pose,
- move to the pre-grasp pose,
- move to the grasp pose,
- close the gripper,
- stabilize the grasp,
- retreat,
- return to the observe pose,
- and call `ai_verify_atomic_task()` as the final completion judgment.

### AI generation time in the demo

In the recorded real demo:

- **TaskAgent** took **15.130 s**
- **ProgramAgent** took **23.570 s**, **27.861 s**, **27.156 s**, and **22.515 s** for the four atomic tasks

Therefore, the **total AI generation time** for this four-step demo was:

- **116.232 s**

We intentionally **do not use total wall-clock execution time as a key metric** in the README, because it includes manual execution and human confirmation latency and therefore does not accurately represent model-side generation efficiency.

### Why this demo matters

This demo shows that the repository already completes the entire first-stage loop:

- complex instruction input,
- automatic decomposition,
- per-step code generation,
- real hardware execution,
- atomic-task completion verification,
- and continuation to the next atomic task.

In other words, the **full head-to-tail chain is already running on a real robot**.

---

## Why the Current Version Still Supports Manual Confirmation

This repository is a **stage-one practical system**, not yet a fully autonomous production stack.

The current design deliberately keeps **manual supervision available** because:

1. the current goal is to validate the end-to-end generation-and-execution pipeline on real hardware;
2. automatic verification is useful, but real-scene perception and execution can still be noisy;
3. manual supervision makes failure localization clearer and keeps debugging costs low.

Therefore, the current second layer should be understood as:

- **Perception**
- **Execution**
- **Confirmation / Verification**

where confirmation may be:

- **human-assisted**, or
- **AI-based through `ai_verify_atomic_task()`**.

This is a deliberate engineering design rather than an accidental inconsistency.

---

## Current Strengths

- Runs on a **real PiPER robot**
- Uses **real RGB-D perception**
- Supports **complex-task decomposition**
- Generates **directly executable code**
- Grounds generation in a **bounded runtime API**
- Preserves **interpretability** via atomic tasks and linear programs
- Supports **manual / review / auto** workflows
- Produces complete logs for each run and each atomic task
- Includes a **generic AI verification entry point** for atomic-task completion

---

## Current Limitations

This repository is a stage-one real-execution prototype, so several limitations are explicit:

1. **Automatic verification is not perfect**  
   Although `ai_verify_atomic_task()` improves automation, complex cluttered scenes can still challenge visual judgment.

2. **Execution is atomic-task sequential**  
   The current implementation advances only after each atomic task is judged complete.

3. **Program generation is constrained but still LLM-dependent**  
   Code quality depends on the current prompt design, runtime API design, and the specific model backend.

4. **Perception remains fragile in visually ambiguous scenes**  
   Similar-colored objects, occlusions, and ambiguous regions remain difficult.

5. **Current task generality is still limited**  
   The present focus is tabletop pick-and-place style manipulation rather than arbitrary open-world robotics.

---

## Relation to the Paper Direction

This repository is not meant to be “just another robotics demo.” It is the engineering realization of the paper idea in its first executable form.

The mapping is direct:

- **Paper idea:** dual-layer decomposition  
  **Repository implementation:** TaskAgent decomposition + atomic-task execution loop

- **Paper idea:** dynamic API-based program generation  
  **Repository implementation:** ProgramAgent generates executable code over `runtime_api`

- **Paper idea:** perception / execution / confirmation  
  **Repository implementation:** grounded perception + executable code + generic AI verification or manual confirmation

- **Paper idea:** use LLMs as structured task-to-program translators rather than end-to-end motor policies  
  **Repository implementation:** Doubao generates executable scripts from the current atomic task and allowed API docs

This makes the repository a practical stepping stone from concept to a fuller embodied-agent framework.

---

## Citation

If you find this repository useful, please cite the project and star the GitHub repository:

```text
https://github.com/zyb45/hpaf
```

---

## Acknowledgement

This repository is inspired by and developed in dialogue with prior work on:

- large-language-model-based robot planning,
- vision-language-action models,
- dynamic program generation for embodied tasks,
- and multi-task / few-shot robot learning.

The goal is not to directly reproduce any single prior system, but to build a practical framework oriented toward:

- **dual-layer task decomposition**,
- **API-grounded code generation**,
- **generic atomic-task verification**,
- and **real robot execution**.

---

## Final Note

HPAF is currently at the stage of **“the first real executable closed loop”** rather than **“the final autonomous embodied system.”**

That is exactly why this repository matters: it already proves that the central chain can run on real hardware:

**complex task -> atomic tasks -> executable code -> robot execution -> completion verification -> next atomic task**

This is the foundation on which stronger perception, richer APIs, more reliable automatic verification, and larger-scale experiments can be built.
