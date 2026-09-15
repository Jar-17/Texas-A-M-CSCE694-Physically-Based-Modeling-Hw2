# Texas A&M CSCE694 — Physically Based Modeling HW1

Starter scaffold for the assignment.

## Setup
```bash
pip install -r requirements.txt
python src/main.py
```

## Controls
- Use the left-side control panel sliders to adjust gravity, x/y wind, air drag, restitution, friction, and physics timestep while the simulation runs.
- Use the panel buttons to reset the ball, reset the camera, pause/resume, or select a launch preset.
- `1`-`4` select repeatable launch cases (side, floor, top, and corner motion)
- `R` reset the current case
- `SPACE` pause/resume
- `UP`/`DOWN` decrease/increase the Euler integration step size
- `LEFT`/`RIGHT` adjust the x-direction wind force
- `F`/`G` increase/decrease collision friction
- `H`/`J` increase/decrease air drag
- `A`/`D` orbit, `W`/`S` tilt, `Q`/`E` zoom, `C` reset the camera
- `ESC` quit

## Implementation notes
- The simulation uses explicit Euler integration for position and velocity.
- Every step performs a swept sphere-vs-box test. If a face is reached before the end of the step, the step is split at that fractional contact time, collision response is applied, and the remaining time is integrated.
- Forces include gravity, constant wind, and velocity-dependent air drag. Collision response includes restitution and tangential friction.
- The container is a full 3D box with six faces. The display is a projected wireframe view so motion in x, y, and z can be observed.
- The render timer runs at 60 FPS independently of the adjustable physics step size. The on-screen `dt` value makes the integration setting visible during a demonstration.

## Demonstration checklist
1. Run `python src/main.py`.
2. Use cases `1`-`4` to show side, floor, top, and multi-axis/corner bounces.
3. Change `dt` with `UP`/`DOWN` while the animation continues at the same display rate.
4. Increase friction with `F` and wind with `LEFT`/`RIGHT`; pause with `SPACE` to point out the current state.

The simulation logic is in `src/physics/world.py`; the Pygame rendering, control panel, and user interface are in `src/main.py`.
