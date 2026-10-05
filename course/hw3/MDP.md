# K1 MDP inventory and teaching decisions

This document inventories the original Flat task and course reward changes.
The original source commit is recorded in [PROVENANCE.md](PROVENANCE.md).
`run.py check --dump` prints the live baseline configuration without
requiring GPU training.

## Runtime and minimization

The `mjlab.tasks` package entry point now explicitly registers only
`Mjlab-Velocity-Flat-Booster-K1`. The lab launcher adds the two course tasks and
lists only those three IDs. mjlab still discovers its own built-in tasks; its
framework is unchanged. The vendor package uses an isolated uv project pinned
to mjlab `b517e0c489139e7fcee95702cfb2b01931264985`, Python 3.13 and uv_build 0.9.x.
Direct requirements are mjlab, SciPy >=1.16.3 and Viser >=1.1.0. The updated
lock has 130 packages; all retained package versions match the original lock.

The Flat factory still calls the Rough factory and common velocity factory,
then selects a plane and removes terrain-level curriculum. Their source is
unchanged. Serial K1 constants and actuators load `xmls/k1.xml` and its 25
meshes. mjlab Entity assembles actuators: nu=22, nq=29, nv=28. PPO uses the
unchanged Adam configuration and BoosterOnPolicyRunner -> mjlab's velocity
runner -> rsl-rl. The exporter attaches Booster metadata after saving.

`run.py train` delegates to the preserved Booster wrapper, narrowed to plain
PPO while retaining YAML serialization cleanup, and then mjlab launch_training.
Playback retains the recording Viser viewer and delegates to mjlab. Shared
symmetry functions remain used for PPO diagnostics despite augmentation/mirror
loss being disabled. AMP, motion/tracking, parallel models, Muon and FastSAC
are disconnected and removed. No retained dependency needs HF/pandas/PyArrow.
Shared terrain/symmetry/extra-pose/export definitions are kept intact because
splitting required modules would add unnecessary risk. Per-file evidence and
original/current hashes are in `vendor-classification.json`.

## Observations

`velocity_env_cfg.py` defines ordered actor/critic groups. Sources below are
in the pinned **installed mjlab**, not copied into Booster:

- **O**: `mjlab/envs/mdp/observations.py`.
- **V**: `mjlab/tasks/velocity/mdp/observations.py`.

| Term | Actor / critic dimensions | Function/source | Meaning and teaching/control assessment |
|---|---:|---|---|
| `base_ang_vel` | 3 / 3 | `builtin_sensor` (O), `robot/imu_ang_vel` | Body IMU angular rates; basic stabilization, realistic gyro signal |
| `projected_gravity` | 3 / 3 | `projected_gravity` (O) | Root orientation's normalized gravity vector in body frame; basic tilt signal, understandable attitude estimate; not directly a raw IMU accelerometer |
| `joint_pos` | 22 / 22 | `joint_pos_rel` (O) | Position relative to default pose; essential proprioception; actor uses `biased=True`, critic ground truth |
| `joint_vel` | 22 / 22 | `joint_vel_rel` (O) | Velocity relative to default (zero here); standard encoder-derived proprioception |
| `actions` | 22 / 22 | `last_action` (O) | Last raw action; policy memory for smoothness and delayed actuation; software-realistic |
| `command` | 3 / 3 | `generated_commands` (O), `twist` | Desired x/y velocity, yaw rate; essential goal conditioning |
| `base_lin_vel` | — / 3 | `builtin_sensor` (O), `robot/imu_lin_vel` | Privileged simulator body linear velocity; hardware requires estimation |
| `foot_height` | — / 2 | `foot_height` (V) | Per-foot terrain clearance from scan; privileged terrain measurement |
| `foot_air_time` | — / 2 | `foot_air_time` (V) | Current time since contact; useful gait state, needs contact estimate on hardware |
| `foot_contact` | — / 2 | `foot_contact` (V) | Boolean contact converted to float; privileged contact state |
| `foot_contact_forces` | — / 6 | `foot_contact_forces` (V) | Two 3-vectors, flattened and `sign(F)*log1p(abs(F))`; privileged forces |

Totals: **actor 75, critic 90, actions 22**, confirmed by runtime manager tables
and reset/step checks. No history stacking is configured. Actor uniform additive
noise: angular rate ±0.2, gravity ±0.05, joint position ±0.01, joint velocity ±1.5.
Critic corruption is disabled; copied angular/gravity noise specifications are
therefore inactive for it. Startup encoder bias is independently ±0.015 radians
and affects the actor joint position, not critic joint position. Neural network
observation normalization is separately enabled for both actor and critic.

Recommendation: retain all observation terms. They are standard locomotion
inputs and critic privileges; deleting them would change the control problem
and model input dimensions without a clear teaching benefit.

## All 15 active rewards

Sources:

- **B (custom)**: `booster_mjlab/tasks/velocity/mdp/rewards.py`.
- **R (built-in)**: `mjlab/tasks/velocity/mdp/rewards.py`.
- **E (built-in)**: `mjlab/envs/mdp/rewards.py`.

Weights are configured in Booster's `velocity_env_cfg.py`; K1 factory fills
robot body/site/geom selections. The reward manager applies weight and default
control timestep scaling (0.02 seconds). Functions return unweighted values.
Risk statements are hypotheses, not performance results.

| Name | Callable / source | Weight | Dependencies and purpose | Complexity / proposed student role | Plausible removal risk |
|---|---|---:|---|---|---|
| `track_linear_velocity` | `track_linear_velocity` B | +2.25 | `twist`, body linear velocity including z; speed-relative exponential error plus progress | Medium; STUDENT_IMPLEMENT | Standing instead of following speed; vertical bouncing |
| `track_angular_velocity` | `track_angular_velocity` R | +2.0 | `twist`, body angular velocity; yaw error plus roll/pitch rates | Low; STUDENT_IMPLEMENT | Failing to turn, excess angular motion |
| `upright` | `variable_upright` B | +1.0 | `twist`, Trunk quaternion/gravity; speed-dependent tilt tolerance | Medium; STUDENT_IMPLEMENT | Leaning/falling or exploiting velocity tracking |
| `body_ang_vel` | `body_angular_velocity_penalty` R | −0.01 | Trunk world angular x/y rates squared | Low; PROVIDED_CORE | More trunk oscillation |
| `angular_momentum` | `angular_momentum_penalty` R | −0.005 | `robot/root_angmom`; whole-body squared angular momentum | Medium; CANDIDATE_REMOVE | Unnatural/excessive arm or body momentum |
| `dof_pos_limits` | `joint_pos_limits` E | −1.0 | Robot joint positions and soft limits | Low; PROVIDED_SAFETY_SHAPING | Joint-limit exploitation or hard-stop behavior |
| `action_rate_l2` | `action_rate_l2` E | −0.1 | Raw current/previous actions; squared change summed over joints | Low; STUDENT_IMPLEMENT | Chattering and abrupt commands |
| `air_time` | `feet_air_time` R | +0.1 | Feet contact air-time, `twist`; reward time in specified window | Medium; STUDENT_IMPLEMENT | Shuffling or weak stepping; term itself can encourage hopping if overweighted |
| `foot_clearance` | `feet_clearance` R | −2.0 | Terrain scan, foot-site xy speed, `twist`; absolute height error weighted by speed | Medium; PROVIDED_SAFETY_SHAPING | Dragging toes or excessive lift |
| `foot_swing_height` | stateful `feet_swing_height` R | −0.25 | Scan, contact/first landing, `twist`; remembered peak-height error at landing | High; CANDIDATE_REMOVE | Worse swing trajectory/foot clearance |
| `foot_slip` | `feet_slip` R | −0.2 | Foot sites, contact, `twist`; xy speed squared while grounded | Medium; STUDENT_IMPLEMENT | Sliding rather than stepping |
| `soft_landing` | `soft_landing` R | −0.001 nominal; curriculum below | Contact force and first-contact timing, `twist`; landing impact magnitude | Medium; PROVIDED_SAFETY_SHAPING | Hard impacts, hardware stress |
| `self_collisions` | `self_collision_cost` R | −1.0 | `self_collision` found slots; sum contact counts | Low concept, complex sensing; PROVIDED_SAFETY_SHAPING | Limbs colliding or interpenetration exploitation |
| `upper_body_posture` | stateful `upper_body_posture_penalty` B | −0.1 | Upper joints/default pose, `twist`; normalized squared posture deviation by speed | High; PROVIDED_CORE | Uncontrolled head/arm pose and reduced hardware acceptability |
| `standing_pose_l1` | `standing_pose_deviation_l1` B | −0.5 | All joints/default pose, `twist`; mean absolute deviation gated toward rest | Low/medium; STUDENT_IMPLEMENT | Moving unnecessarily when commanded to stand |

Exact parameter highlights (remaining defaults are in the JSON snapshot):

- Linear tracking: `sigma=clamp(0.75*norm(cmd_xy), 0.1, 0.5)`.
  Exponential uses xy-error squared **plus vertical velocity squared**;
  `progress_weight=0.5` blends it with clamped normalized planar progress.
- Angular tracking: `std=sqrt(0.5)`; exponential includes commanded yaw-rate
  error and uncommanded roll/pitch-rate squares. Its source docstring mentions
  heading error, but its implementation evaluates rates; heading commands are
  converted to yaw-rate commands by the command generator.
- Upright: standing/walking/running `std=sqrt(0.20/0.25/0.35)` with speed
  thresholds 0.05 and 1.5. Speed is `norm(cmd_xy)+abs(cmd_yaw)`, mixing physical
  units; explain this heuristic rather than calling it a pure linear speed.
- Air time: per-foot reward when current air time is between 0.05 and 0.5 s;
  only active for total command >0.2. It is not just a touchdown reward.
- Clearance/swing/slip/landing: command gate >0.05. Target height 0.06 m.
  Clearance uses absolute height error times xy foot speed. Swing penalty uses
  squared normalized **peak** height error at first contact; it stores state.
  Slip uses grounded xy foot speed squared. Landing uses force norm at contact.
- Upper posture: selects `Head_.*`, `.*_Shoulder_.*`, `.*_Elbow_.*`; standing
  std 0.05 for all; walking head 0.05, shoulders/elbows 0.15; running head 0.05,
  shoulder pitch 0.5, shoulder roll 0.2, elbows 0.35. Thresholds 0.05 and 1.0.
- Standing pose: `mean(abs(q-q_default)) * (1-clamp(total_command/0.05,max=1))`.
- Soft joint bounds use factor 0.9; penalties are for exceeding those soft
  limits, not an extra termination condition.

The upright measure uses projected-gravity XY magnitude; it alone cannot
distinguish upright from completely inverted. Retained orientation/contact
terminations and the other shaping terms matter.

The course task replaces seven terms with the explicitly specified course
formulation below. It intentionally changes rewards, not the other MDP or PPO
settings. The 13-term candidate removes only angular momentum and peak swing
height. Upper-body posture remains provided. No removal is considered
REMOVE_IF_VALIDATED yet: that decision requires GPU policy comparisons.

## Commands, terminations and curricula

`twist` uses Booster's thin subclass of mjlab `UniformVelocityCommand`.
Resample uniformly every 1–4 s; 20% standing, 30% heading-controlled, 10%
forward-only assignments (these selections can overlap). Heading targets are
uniform in [−π,π]; yaw command uses heading stiffness 0.5 and rate clipping.
Forward-only assignments set positive x speed with minimum 0.3 and zero y/yaw
at sampling; heading update and standing overrides apply afterward. World-frame
fraction and command-driven initial-velocity probability are both zero.
Initial configured ranges are x/y/yaw [−0.5,0.5]; curriculum replaces them.

| Termination | Source | Behavior |
|---|---|---|
| `time_out` | mjlab env MDP `time_out` | 20 s, 1,000 control steps; marked time-out for bootstrapping |
| `fell_over` | Booster `mdp/terminations.py:stochastic_bad_orientation` | Each step beyond 63° tilt terminates independently with probability 0.02; no countdown |
| `illegal_contact` | mjlab velocity `illegal_contact` | Non-foot contact sensor detects forbidden ground contact |

Flat removes only the terrain-level curriculum. Remaining stages use absolute
`common_step_counter` values, **not individual transitions across all envs**:

| Control steps | x range | y range | yaw-rate range |
|---:|---|---|---|
| 0 | [−1.0,1.2] | [−1.0,1.0] | [−1.0,1.0] |
| 120,000 (5,000×24) | [−1.0,1.5] | [−1.25,1.25] | [−1.25,1.25] |
| 240,000 | [−1.25,1.5] | [−1.5,1.5] | [−1.5,1.5] |
| 360,000 | [−1.5,1.75] | [−1.75,1.75] | [−1.5,1.5] |

Landing weight stages: −0.0001 at step 0; −0.001 at 24,000; −0.005 at
168,000. Hence the configured nominal −0.001 is not the whole training story.
The curriculum runs on reset, so stage changes take effect at those updates.
Resume state restores the common step counter; do not reset stages by changing
rollout length casually. No custom metric terms are configured, but reward and
command functions log diagnostic metrics.

## Events and domain randomization

| Name / mode | Source | Parameters / affected state |
|---|---|---|
| `terrain_contact` / startup | Booster `mdp/terrain.py:randomize_terrain_contact` | Terrain solref index0 [0.006,0.03], index1 [0.95,1.05]; solimp index0 [0.88,0.92], index1 [0.94,0.99], index2 [0.003,0.01]; shared random draw |
| `foot_friction` / startup | mjlab `mdp.dr.geom_friction` | Absolute foot geom friction [0.75,1.25]; both feet share draw |
| `encoder_bias` / startup | mjlab `mdp.dr.encoder_bias` | All robot joints ±0.015 rad |
| `pd_gains` / startup | mjlab `mdp.dr.pd_gains` | All actuator kp and kd scaled [0.8,1.2] |
| `trunk_inertia` / startup | mjlab `mdp.dr.pseudo_inertia` | Trunk alpha/t ranges ±0.05 |
| `limb_inertia` / startup | mjlab `mdp.dr.pseudo_inertia` | Every non-Trunk body; alpha ±0.05, t ±0.025 |
| `reset_base` / reset | mjlab `reset_root_state_uniform` | Pose offsets x/y ±0.5 m, z [0.01,0.05], yaw ±3.14; velocity x/y ±1, z [0.01,0.3], roll/pitch ±0.1, yaw ±0.5 |
| `reset_robot_joints` / reset | mjlab `reset_joints_by_offset` | Position and velocity offsets ±0.1 |
| `push_robot` / interval | mjlab `push_by_setting_velocity` | Every 1.5–4 s; velocity x/y ±0.28, z ±0.2, roll/pitch ±0.52, yaw ±0.78 |

Pseudo-inertia parameters are the implementation's physically valid transform
parameters, not independent percentage changes to each inertia component. Keep
these and all other randomization provided; students need not implement them.

## Sensors, robot, terrain and physics

- XML built-ins include `imu_lin_vel`, `imu_ang_vel`, `imu_lin_acc`, orientation
  sensors and `root_angmom`; active observations/rewards use the velocity and
  angular-momentum sensors listed above. Unused built-ins remain intact.
- `feet_ground_contact`: foot subtrees versus terrain, found/force fields,
  net force reduction, one slot, air-time tracking. Feeds critic and foot rewards.
- `non_foot_ground_contact`: robot bodies excluding foot links versus terrain,
  found/force, net force reduction. Feeds illegal-contact termination.
- `self_collision`: Trunk subtree versus itself, found field, no reduction,
  one slot. Feeds collision penalty.
- `foot_height_scan`: custom `FootClearanceSensorCfg`, two foot sole sites,
  5×3 sample grid over 0.13×0.06 m per foot. Measures terrain clearance with
  vertical queries, clamps observed height at max distance 1 m; includes
  terrain group0 and excludes parent body. Configured yaw alignment does not
  rotate query direction in this custom sensor. Feeds critic and clearance/
  peak-height rewards. Keep it even though Flat has no rough terrain.
- Flat plane, terrain generator `None`; robot full collisions and shared terrain
  compliance/friction conventions remain as configured. Foot contact geoms,
  foot reward sites and Trunk body are filled by the robot-specific factory.
- Joint-position actions cover all actuators, use default offsets, and scales
  `0.25*effort_limit/stiffness` by actuator group. Configured delays: min lag2,
  max lag8, hold probability0.3. Default pose from `HOME_KEYFRAME`; root z0.5125 m.
- MuJoCo timestep0.005 s, decimation4, solver iterations10, line-search20;
  Flat `njmax=300`, `nconmax=50`, `contact_sensor_maxmatch=64`, CCD iterations60.

## PPO and play differences

Actor and critic: ELU MLP [512,256,128], normalized inputs. Actor Gaussian initial
std1.0; plain PPO with Adam, LR0.001/adaptive, 5 epochs, 4 minibatches, clip0.2,
value coefficient1, clipped value loss, entropy0.01, gamma0.99, lambda0.95,
desired KL0.01, gradient norm1. Rollout24, default30,000 iterations, save50,
seed42, W&B project `mjlab`, experiment `k1_velocity`. Neither AMP discriminator,
DA optimization nor Muon is active. Symmetry diagnostic callback remains.

Play factory: episode length1e9 s, actor corruption disabled, illegal-contact
termination removed; pushes remain enabled. Command ranges broaden to
x±2.5, y±2.0, yaw±3.7, although the retained reset-time command curriculum can
replace these ranges. Domain randomization and the other termination/curriculum
terms remain. Use matched train-mode command conditions for quantitative
comparisons rather than assuming viewer defaults are a controlled evaluation.

## Seven course reward implementations

All callables below are in `course_booster/rewards.py`. Parameters and nominal
weights are in `course_booster/env_cfg.py`. Each function returns an unweighted
batched tensor. Each function provides the state access and a TODO describing
the math; there is no forwarding call to an existing reward helper.

| Course term (baseline key) | Weight | Parameters and physical inputs | Purpose / workload / risks |
|---|---:|---|---|
| `track_linear_velocity` | +2.25 | Body xyz velocity, twist; longitudinal std 0.1–0.5, relative 0.75, transverse 0.3, vertical 0.5, progress blend 0.5 | Direction-aware tracking; medium tensor/projection work; reward scale changes and progress may dilute vertical suppression |
| `track_angular_velocity` | +2 | Body yaw rate, twist; variance 0.5 | Isolate commanded turning; low mathematical complexity; no roll/pitch suppression inside this term, retained body angular shaping remains important |
| `upright` | +1 | Normalized root projected gravity | Signed alignment distinguishes inversion; low computation, physical frame reasoning; speed-dependent leaning tolerance is intentionally removed |
| `action_smoothness` (`action_rate_l2`) | -0.1 | Current/previous/two-step-old raw action; acceleration weight 0.1 | Rate plus acceleration regularization; medium temporal indexing work; may suppress responsive control if overweighted |
| `standing_pose` (`standing_pose_l1`) | -0.5 | Joint/default positions and velocities; separate command gates 0.05 m/s and 0.05 rad/s, velocity weight 0.01 | Quiet standing; medium broadcast/reduction/gating work; L2 mean scale differs from baseline L1 |
| `air_time` | +0.1 | Contacts and current air times; triangular 0.05–0.5 s window; separate movement thresholds 0.2 | Gait/contact masking; medium work; sum <=2, less total reward than rectangular window; can still incentivize hopping |
| `foot_slip` | -0.2 | Matched foot-site world xy velocities and grounded mask | Contact-only slip penalty, also active standing; medium frame/mask work; may inhibit corrective foot movements if overweighted |

Course upright uses root gravity; the original Trunk is the root. Reward
calculations access physical state rather than the actor's noisy observations.
The second action-history tensor is already available in the manager, but the
feedforward actor observes only the previous action: this regularizer uses
hidden history. We do not expand observations to accommodate it; keep its
effect in mind when tuning. Gait window widths/thresholds must
be positive and ordered; all scale parameters must be positive. Default config
is valid; invalid student parameter choices need fixing before training.

The upstream air-time reward logs `Metrics/air_time_mean` as a side effect.
The course reward returns its tensor only, so that diagnostic is absent; do not
use it as a cross-task comparison metric without a separate evaluator.

## Provided shaping and candidate

Provided core: `body_ang_vel`, `upper_body_posture`. Provided safety shaping:
`dof_pos_limits`, `foot_clearance`, `soft_landing`, `self_collisions`. Candidate
removal terms: `angular_momentum`, `foot_swing_height`. All eight remain active
in the full course reference; only the latter two are omitted in the 13-term
candidate. Sensors remain present even when a candidate no longer uses one
for reward; they also support critic observations and the unchanged baseline.

The candidate tests whether whole-body momentum and remembered peak swing
height can be omitted while clearance/contact/impact/posture shaping remains.
Potential regressions include arm momentum, foot trajectories and impacts.
Do not attribute a difference between upstream and the course reference to
pruning: seven reward formulas intentionally differ. Use upstream parity for
infrastructure and reference-versus-candidate training for the two-term ablation.

The public upstream/mjlab sources retain related rewards because the original
baseline and provided shaping need them. They are related techniques, not
copies of the course formulas; implement yours from the specification.
CPU reset/PPO/export success does not establish policy quality.
