# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), 
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.26.0] - 2025-12-01 11:20:56

### Fixed

- LA calibration now consistent with constituency calibration.

## [1.25.0] - 2025-11-29 00:33:46

### Added

- Student loan plan imputation based on age and reported repayments

## [1.24.2] - 2025-11-28 17:23:26

### Fixed

- Fix changelog encoding test to skip when changelog_entry.yaml is empty after versioning

## [1.24.1] - 2025-11-28 16:14:35

### Changed

- Calibrate savings income from ONS National Accounts D.41g household interest data instead of SPI (fixes underestimation from ~3bn to ~55bn)

## [1.24.0] - 2025-11-27 18:33:27

### Added

- rail_usage variable derived from rail_subsidy_spending / fare_index at survey year, enabling fare reforms to modify prices independently of usage quantity

## [1.23.3] - 2025-11-27 17:11:09

### Fixed

- Updated personal allowance reform test expected value after calibration fix.

## [1.23.2] - 2025-11-27 16:29:01

### Added

- SS HMRC calibration targets.

## [1.23.1] - 2025-11-27 16:12:15

### Fixed

- Hallucinated calibration targets.

## [1.23.0] - 2025-11-27 15:30:00

### Added

- Add salary sacrifice NI relief as calibration targets (employee £1.2bn, employer £2.9bn from SPP)

## [1.22.0] - 2025-11-26 22:46:41

### Added

- Salary sacrifice imputation using FRS SALSAC routing question to impute ~30% employee participation per HMRC survey data.

## [1.21.0] - 2025-11-20 13:08:22

### Added

- Add pension_contributions_via_salary_sacrifice variable from FRS SPNAMT field

## [1.20.0] - 2025-10-21 12:09:14

### Added

- Universal Credit calibration at national level by award amount and family type, and at constituency level in total.

## [1.19.6] - 2025-10-21 10:18:38

### Changed

- Refactored income imputation to selectively impute only dividend income on the main dataset.
- Removed winter fuel allowance from loss calculations.

## [1.19.5] - 2025-10-20 16:37:14

### Fixed

- Bump patch version to try and get HF upload passing.

## [1.19.4] - 2025-10-07 16:32:10

### Added

- Regional and country labels for UK constituencies.

## [1.19.3] - 2025-10-02 16:46:59

### Changed

- Relaxed childcare test tolerance to allow ratios within 100% of target (0 to 2.0)

## [1.19.2] - 2025-10-02 16:12:23

### Changed

- Relaxed childcare test tolerance to allow ratios up to 1.6

## [1.19.1] - 2025-10-02 15:18:04

### Changed

- Remove birth_year from FRS dataset generation to allow dynamic calculation

## [1.19.0] - 2025-10-02 14:29:16

### Fixed

- Re-add dividends to calibration target set.

## [1.18.0] - 2025-09-30 13:58:18

### Changed

- Upgraded documentation to Jupyter Book 2.0 (MyST-based)

### Fixed

- Jupyter Book deployment to GitHub Pages by adding docs workflow and fixing branch reference

## [1.17.11] - 2025-09-11 16:14:02

### Fixed

- Add is_married to FRS benefit unit dataset

## [1.17.10] - 2025-09-01 09:33:20

### Added

- Calibration to benefit cap statistics.

## [1.17.9] - 2025-08-20 10:32:56

### Fixed

- Imputation model syntax.

## [1.17.8] - 2025-08-06 10:28:59

### Fixed

- Test result.

## [1.17.7] - 2025-08-06 10:08:10

### Changed

- Moved to functional, simplified architecture.

## [1.17.6] - 2025-08-05 11:47:08

### Added

- Add index.yaml to fix GitHub Pages 404 error

## [1.17.5] - 2025-08-05 11:15:17

### Added

- Added index.html file to docs folder to fix GitHub Pages 404 error

## [1.17.4] - 2025-08-04 15:08:06

### Fixed

- Migrated to more efficient dataset version.

## [1.17.3] - 2025-07-22 11:17:39

### Fixed

- UK model actually bumped.

## [1.17.2] - 2025-07-22 09:41:46

### Changed

- Updated policyengine-uk to 2.40.2 (pin).

## [1.17.1] - 2025-07-22 08:53:18

### Fixed

- Bug in new multi-year dataset.

## [1.17.0] - 2025-07-21 13:53:43

### Added

- New multi-year dataset format for FRS and Enhanced FRS.

## [1.16.2] - 2025-07-17 11:44:26

### Added

- Council Tax calibration.

## [1.16.1] - 2025-07-14 15:21:27

### Fixed

- Added calibrated weights from 2022.

## [1.16.0] - 2025-07-10 16:01:46

### Added

- Structural insurance payments
- External child payments
- Healthy Start payments

## [1.15.3] - 2025-07-10 12:41:13

### Fixed

- SSMG uprating.

## [1.15.2] - 2025-07-09 15:25:53

### Fixed

- Free school meals magnitude error.

## [1.15.1] - 2025-07-09 14:19:38

### Added

- Free school meals

## [1.15.0] - 2025-06-27 09:15:13

### Added

- PIP calibration.

## [1.14.4] - 2025-06-24 13:14:48

### Fixed

- Missing columns in the calibration log.

## [1.14.3] - 2025-06-24 11:39:46

### Fixed

- Name corrected in calibration build artifact.

## [1.14.2] - 2025-06-24 11:18:42

### Added

- Calibration improvements.

## [1.14.1] - 2025-06-24 10:48:32

### Added

- Calibration log exporting.

## [1.14.0] - 2025-06-19 21:08:19

### Added

- Public service imputations.

## [1.13.3] - 2025-06-16 15:00:47

### Fixed

- Documentation used 2022 datasets rather than 2025.

## [1.13.2] - 2025-06-13 14:51:39

### Fixed

- Documentation publishes.
- Local authority calibration consistent with constituency calibration.
- Domestic rates are nonzero.

## [1.13.1] - 2025-06-10 12:41:53

### Fixed

- Documentation deployment.

## [1.13.0] - 2025-06-10 12:37:20

### Changed

- Tax-benefit targets updated from new DWP forecasts.

## [1.12.0] - 2025-06-09 20:25:41

### Fixed

- Inconsistent local area targets removed.

## [1.11.6] - 2025-05-27 14:40:08

### Fixed

- Uprating in child limit target.

## [1.11.5] - 2025-05-27 13:30:04

### Added

- Child limit affected household calibration.

## [1.11.4] - 2025-05-26 22:10:54

### Fixed

- Added missing HF token

## [1.11.3] - 2025-05-26 21:44:28

### Fixed

- Typo in huggingface repo location.

## [1.11.2] - 2025-05-26 21:38:53

### Fixed

- GCP uploads to buckets happen alongside HF uploads.
- Versioning in data uploads.

## [1.11.1] - 2024-12-10 10:46:07

### Fixed

- Documentation errors.

## [1.11.0] - 2024-12-09 11:51:05

### Added

- Local authority weights.

## [1.10.1] - 2024-12-03 17:25:35

### Added

- Dropout in constituency calibration.

## [1.10.0] - 2024-12-03 11:21:54

### Added

- Target uprating for constituencies.

## [1.9.2] - 2024-11-30 13:23:17

### Fixed

- Constituency weights are in A-Z order.

## [1.9.1] - 2024-11-27 19:28:29

### Added

- Automatic calibration.

## [1.9.0] - 2024-10-22 11:18:48

### Fixed

- Bug removing capital gains.

## [1.8.0] - 2024-10-22 08:30:52

### Changed

- Data URLs updated.

## [1.7.0] - 2024-10-21 17:03:50

### Added

- Calibration for private school students.

## [1.6.0] - 2024-10-18 16:05:10

### Added

- Future year income targeting.
- Random takeup variable values.

## [1.5.0] - 2024-10-16 17:05:58

### Added

- Moved epoch count to 10k per year.

## [1.4.0] - 2024-10-16 17:05:39

### Added

- Missing changelog entry.

## [1.3.0] - 2024-10-16 17:02:56

### Added

- Re-run calibration with more epochs.

## [1.2.5] - 2024-09-18 13:57:40

### Fixed

- GH actions naming.
- Bug causing the Extended FRS to error.

## [1.2.4] - 2024-09-18 13:26:36

## [1.2.3] - 2024-09-18 12:57:28

### Fixed

- Bug causing the Extended FRS to not generate.

## [1.2.2] - 2024-09-18 11:38:21

## [1.2.1] - 2024-09-18 10:05:45

### Fixed

- Data download URLs.

## [1.2.0] - 2024-09-18 00:32:05

### Fixed

- Compatibility with PolicyEngine UK.

## [1.1.0] - 2024-09-17 18:05:27

### Changed

- Lightened dependency list.

## [1.0.0] - 2024-09-09 17:29:10

### Added

- Initialized changelogging



[1.26.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.25.0...1.26.0
[1.25.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.24.2...1.25.0
[1.24.2]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.24.1...1.24.2
[1.24.1]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.24.0...1.24.1
[1.24.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.23.3...1.24.0
[1.23.3]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.23.2...1.23.3
[1.23.2]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.23.1...1.23.2
[1.23.1]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.23.0...1.23.1
[1.23.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.22.0...1.23.0
[1.22.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.21.0...1.22.0
[1.21.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.20.0...1.21.0
[1.20.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.19.6...1.20.0
[1.19.6]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.19.5...1.19.6
[1.19.5]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.19.4...1.19.5
[1.19.4]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.19.3...1.19.4
[1.19.3]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.19.2...1.19.3
[1.19.2]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.19.1...1.19.2
[1.19.1]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.19.0...1.19.1
[1.19.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.18.0...1.19.0
[1.18.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.17.11...1.18.0
[1.17.11]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.17.10...1.17.11
[1.17.10]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.17.9...1.17.10
[1.17.9]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.17.8...1.17.9
[1.17.8]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.17.7...1.17.8
[1.17.7]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.17.6...1.17.7
[1.17.6]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.17.5...1.17.6
[1.17.5]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.17.4...1.17.5
[1.17.4]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.17.3...1.17.4
[1.17.3]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.17.2...1.17.3
[1.17.2]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.17.1...1.17.2
[1.17.1]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.17.0...1.17.1
[1.17.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.16.2...1.17.0
[1.16.2]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.16.1...1.16.2
[1.16.1]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.16.0...1.16.1
[1.16.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.15.3...1.16.0
[1.15.3]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.15.2...1.15.3
[1.15.2]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.15.1...1.15.2
[1.15.1]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.15.0...1.15.1
[1.15.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.14.4...1.15.0
[1.14.4]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.14.3...1.14.4
[1.14.3]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.14.2...1.14.3
[1.14.2]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.14.1...1.14.2
[1.14.1]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.14.0...1.14.1
[1.14.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.13.3...1.14.0
[1.13.3]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.13.2...1.13.3
[1.13.2]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.13.1...1.13.2
[1.13.1]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.13.0...1.13.1
[1.13.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.12.0...1.13.0
[1.12.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.11.6...1.12.0
[1.11.6]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.11.5...1.11.6
[1.11.5]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.11.4...1.11.5
[1.11.4]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.11.3...1.11.4
[1.11.3]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.11.2...1.11.3
[1.11.2]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.11.1...1.11.2
[1.11.1]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.11.0...1.11.1
[1.11.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.10.1...1.11.0
[1.10.1]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.10.0...1.10.1
[1.10.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.9.2...1.10.0
[1.9.2]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.9.1...1.9.2
[1.9.1]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.9.0...1.9.1
[1.9.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.8.0...1.9.0
[1.8.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.7.0...1.8.0
[1.7.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.6.0...1.7.0
[1.6.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.5.0...1.6.0
[1.5.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.4.0...1.5.0
[1.4.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.3.0...1.4.0
[1.3.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.2.5...1.3.0
[1.2.5]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.2.4...1.2.5
[1.2.4]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.2.3...1.2.4
[1.2.3]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.2.2...1.2.3
[1.2.2]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.2.1...1.2.2
[1.2.1]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.2.0...1.2.1
[1.2.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.1.0...1.2.0
[1.1.0]: https://github.com/PolicyEngine/policyengine-us-data/compare/1.0.0...1.1.0
