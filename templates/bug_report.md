# Bug Report Template

## Bug Information
- **Bug ID**: {{bug_id}}
- **Date Created**: {{date_created}}
- **Reporter**: {{reporter}}
- **Priority**: {{priority}}
- **Status**: {{status}}
- **Component**: {{component}}

## Summary
{{summary}}

## Description
{{description}}

## Steps to Reproduce
{{#steps_to_reproduce}}
1. {{.}}
{{/steps_to_reproduce}}

## Expected Behavior
{{expected_behavior}}

## Actual Behavior
{{actual_behavior}}

## Test Information
- **Test File**: {{test_file}}
- **Test Method**: {{test_method}}
- **Test Command**: {{test_command}}

## Error Details
```
{{error_message}}
```

## Stack Trace
```
{{stack_trace}}
```

## Environment
- **Python Version**: {{python_version}}
- **OS**: {{operating_system}}
- **Git Commit**: {{git_commit}}

## Investigation Notes
{{investigation_notes}}

## Potential Solutions
{{#potential_solutions}}
- {{.}}
{{/potential_solutions}}

## Related Issues
{{#related_issues}}
- {{.}}
{{/related_issues}}

## Acceptance Criteria
{{#acceptance_criteria}}
- [ ] {{.}}
{{/acceptance_criteria}}

---
*This bug was generated from failing test: {{test_file}}::{{test_method}}*