# Bug Report

## Bug Information
- **Bug ID**: OBJ-001
- **Date Created**: 2024-01-10
- **Reporter**: Claude Code Assistant
- **Priority**: Medium
- **Status**: Open
- **Component**: ContainerManager

## Summary
ContainerManager.apply_container_effects() failing with KeyError for 'effect_type'

## Description
The apply_container_effects method in ContainerManager is returning a result dictionary that doesn't contain the expected 'effect_type' key, causing tests to fail when trying to access this key.

## Steps to Reproduce
1. Create a ContainerManager instance
2. Create a container with basic specification
3. Call apply_container_effects() with "preservation" effect type
4. Access result["effect_type"]

## Expected Behavior
The method should return a dictionary containing an 'effect_type' key with the value of the effect type that was applied.

## Actual Behavior
KeyError is raised when trying to access result["effect_type"] because the key doesn't exist in the returned dictionary.

## Test Information
- **Test File**: tests/unit/core/containers/test_container_manager.py
- **Test Method**: TestContainerManager::test_apply_container_effects_preservation
- **Test Command**: `poetry run python -m pytest tests/unit/core/containers/test_container_manager.py::TestContainerManager::test_apply_container_effects_preservation -v`

## Error Details
```
KeyError: 'effect_type'
```

## Stack Trace
```
tests/unit/core/containers/test_container_manager.py:XXX: in test_apply_container_effects_preservation
    assert result["effect_type"] == "preservation"
E   KeyError: 'effect_type'
```

## Environment
- **Python Version**: 3.12.4
- **OS**: Linux
- **Git Commit**: Current working directory

## Investigation Notes
The apply_container_effects method appears to be missing the 'effect_type' field in its return dictionary. The method should include this field to match the expected API contract.

## Potential Solutions
- Add 'effect_type' field to the return dictionary in apply_container_effects method
- Ensure all code paths in the method include this field
- Update method documentation to clarify return structure

## Related Issues
- May be related to other container effect methods

## Acceptance Criteria
- [ ] apply_container_effects() returns dictionary with 'effect_type' key
- [ ] 'effect_type' value matches the input effect_type parameter
- [ ] Test test_apply_container_effects_preservation passes
- [ ] No regression in other container effect tests

---
*This bug was generated from failing test: tests/unit/core/containers/test_container_manager.py::TestContainerManager::test_apply_container_effects_preservation*