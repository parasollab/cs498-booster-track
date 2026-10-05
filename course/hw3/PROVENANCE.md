# Booster source provenance and redistribution blocker

Source: https://github.com/IntelligentRoboticsLab/booster_mjlab.git

Original commit: `d92e11395d961096d639446c0a8a176f12bb51da`.

The original snapshot had 193 tracked files / 42,446,543 bytes. This minimized
snapshot retains 63 files / 12,532,241 bytes; 130 files were removed after import,
entry-point, XML/build-reference, configuration, CPU execution and export checks.
`vendor-manifest.json` preserves every original SHA-256. `vendor-classification.json`
records disposition, evidence, original/current hashes and retained sizes.
Original files remain recoverable from the recorded upstream commit.

Patches narrow eager registrations/imports to serial Flat K1 and Adam PPO,
remove unsupported research configuration/console surfaces, and reduce/re-lock
installation requirements. The Flat/Rough/common environment factories, robot
XML/actuators, observations, original rewards and active PPO settings remain
unchanged. The course reward layer is separate. Runtime-dependent shared modules
retain co-located unused definitions instead of invasive line-level pruning.

The upstream README remains verbatim for attribution/history and contains
research commands/media references absent from this snapshot. Students should
use the course handout, not those historical instructions. Upstream author
metadata names Harold Ruiter; original source comments/attribution remain.
There is no nested Git repo, submodule or vendored dependency environment in
course source. The local `.venv` and generated artifacts are ignored.

**STUDENT RELEASE BLOCKER: no top-level software LICENSE/COPYING exists in the
original source snapshot.** `website/static/css/BULMA-LICENSE.txt` is retained
verbatim but licenses Bulma, not Booster software or robot assets. No Booster
license is inferred or invented. Obtain explicit redistribution permission or
appropriate upstream licensing before publishing to students.

Course HW0/HW2 pin mjlab `8ee51fbcf806a7419189f706d9e394cbeb7790fa`; Booster pins
`b517e0c489139e7fcee95702cfb2b01931264985`. Keep environments isolated.
Third-party dependencies are installed by uv, not copied into repository source.
