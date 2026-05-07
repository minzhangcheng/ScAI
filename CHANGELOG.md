# Changelog

## [0.22.8] - 2025-05-07

### Added
- Added `SESSION_AUTH_OPTIONAL_SUFFIXES` tuple in auth middleware to define auth-optional paths under `/api/session/<uuid>/` (e.g., `/report/`, `/export`, `/figure/`)

### Changed
- Refactored auth-optional path checking logic in `AuthMiddleware` to use new `_is_auth_optional()` helper function, improving maintainability and extensibility

---

## [0.22.7] - 2025-04-16

### Added
- Added optional admin initialization check to avoid exposing uninitialized system

---

## [0.22.6] - 2025-04-08

### Fixed
- Fixed session import path resolution issue

---

## [0.22.5] - 2025-04-01

### Added
- Added multi-file session export support

---

## [0.22.4] - 2025-03-25

### Changed
- Improved session storage handling

---

## [0.22.3] - 2025-03-18

### Fixed
- Fixed various bugs

---

## [0.22.2] - 2025-03-10

### Added
- Added new features

---

## [0.22.1] - 2025-03-01

### Added
- Initial release of version 0.22.x series