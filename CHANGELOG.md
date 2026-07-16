# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.3] - 2026-07-16

### Changed

- Replaced the combined `dones` transition field with explicit `terminated`
  and `truncated` fields throughout runners, replay buffers, experiments, and
  learner APIs.

### Fixed

- Fixed DQN, SAC, actor-critic, and PPO bootstrapping at Gymnasium time-limit
  truncations. Truncated episodes are still reset and reported as complete,
  but only true terminations now disable value bootstrapping.

## [0.2.2] - 2026-07-13

### Added

- Added `WarmupController` for sampling random actions before switching to the
  configured controller.
- Added `learning_starts` and `warmup_steps` options to
  `OffPolicyExperimentConfig`.
- Added optional actor regularization to `SACLearner`.

### Changed

- Delayed off-policy updates until the configured number of environment steps
  has been collected.
- Renamed the Gaussian and stochastic controller modules while preserving their
  public imports.
- Expanded algorithm documentation and experiment examples.

## [0.2.1] - 2026-07-06

### Changed

- Updated the package publishing workflow.

## [0.2.0] - 2026-07-06

### Added

- Added on-policy REINFORCE, actor-critic, and PPO learners.
- Added off-policy DQN and SAC learners.
- Added continuous-action support, including Gaussian controllers and replay
  buffer handling.
- Added reusable on-policy and off-policy experiment classes.
- Added runnable examples for the supported algorithms.
- Added `uv`-based development setup instructions.

### Changed

- Reorganized learners and controllers around the on-policy and off-policy
  APIs.
- Improved off-policy batch handling and SAC update efficiency.

### Fixed

- Fixed handling of the last completed episode in off-policy experiments.

## [0.1.1] - 2026-05-31

### Added

- Added the automated PyPI publishing workflow.
- Added repository ownership metadata and expanded documentation.

## [0.1.0] - 2026-05-30

### Added

- Initial release of `drlab`.

[Unreleased]: https://github.com/tomasosarte/drlab/compare/v0.2.3...HEAD
[0.2.3]: https://github.com/tomasosarte/drlab/compare/daff97e...v0.2.3
[0.2.2]: https://github.com/tomasosarte/drlab/compare/ea2820f...daff97e
[0.2.1]: https://github.com/tomasosarte/drlab/compare/4fbd134...ea2820f
[0.2.0]: https://github.com/tomasosarte/drlab/compare/v0.1.1...4fbd134
[0.1.1]: https://github.com/tomasosarte/drlab/releases/tag/v0.1.1
[0.1.0]: https://github.com/tomasosarte/drlab/tree/af34878
