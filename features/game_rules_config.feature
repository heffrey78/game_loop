Feature: Game Loop Rules Configuration
  As a game developer
  I want to configure and manage the game rules system
  So that player interactions with the world are consistent and engaging

  Background:
    Given the game has been initialized
    And the database schema includes the world_rules table

  Scenario: Load static rules from configuration file
    Given a game rules configuration file exists
    When the rules engine initializes
    Then the system should load all static rules from the configuration file
    And the rules should be available to the game engine
    And the rules should be validated for consistency

  Scenario: Configure rules via command line
    When I start the game with command line parameters:
      | parameter              | value       |
      | --enable-custom-rules  | true        |
      | --rules-file           | custom.yaml |
      | --strict-mode          | false       |
    Then the game rules engine should enable custom rules
    And the rules engine should load rules from "custom.yaml"
    And strict validation mode should be disabled

  Scenario: Configure rules file structure
    Given a rules configuration file with sections:
      | section              |
      | movement_rules       |
      | interaction_rules    |
      | combat_rules         |
      | crafting_rules       |
      | world_physics_rules  |
    When the rules engine loads the configuration
    Then each section should be processed independently
    And rule conflicts between sections should be detected
    And the system should log the loaded rule count for each section

  Scenario: Configure dynamic rules system
    Given the game configuration has dynamic_rules set to enabled
    When a player creates a new rule
    Then the system should validate the rule for consistency
    And the validated rule should be added to the world_rules table
    And the rule should be assigned a unique identifier
    And the rule should be active in the current game session

  Scenario: Rules dependency resolution
    Given the following rules exist:
      | rule_id | rule_name         | depends_on     |
      | 1       | gravity           | null           |
      | 2       | water_physics     | gravity        |
      | 3       | fire_propagation  | water_physics  |
    When the rules engine initializes
    Then the system should load rules in the correct dependency order
    And rules with unmet dependencies should be deactivated
    And the system should log any dependency resolution issues

  Scenario: Configure rule priority levels
    When I configure rule priority levels:
      | level | description       | examples                  |
      | 1     | Physics laws      | gravity, conservation     |
      | 2     | World laws        | magic systems, time flow  |
      | 3     | Regional rules    | kingdom laws, zone effects|
      | 4     | Local rules       | building rules, customs   |
      | 5     | Temporary effects | spells, weather effects   |
    Then the rules engine should apply rules in priority order
    And higher priority rules should override lower priority rules
    And conflicts within the same priority should be logged

  Scenario: Configure rule categories
    When I define rule categories:
      | category      | description                    |
      | physical      | how objects interact physically|
      | social        | how NPCs interact socially     |
      | magical       | how magic functions            |
      | environmental | how environment behaves        |
    Then the rules engine should tag rules with appropriate categories
    And rules should be filterable by category
    And rule interactions across categories should be properly managed

  Scenario: Test rule validation system
    Given a set of conflicting rules:
      | rule_name       | rule_effect                   |
      | objects_float   | All objects ignore gravity    |
      | objects_fall    | All objects fall with gravity |
    When I validate the ruleset
    Then the system should identify the conflict
    And the system should suggest resolution strategies
    And the system should require admin confirmation to proceed

  Scenario: Configure rule application scope
    When I define rule scopes:
      | scope       | application_level      |
      | global      | entire game world      |
      | regional    | specific regions       |
      | local       | specific locations     |
      | entity      | specific entities      |
      | player      | specific players       |
    Then rules should only apply within their defined scope
    And the system should track rule boundaries correctly
    And entering or leaving a scope should trigger rule re-evaluation

  Scenario: Configure player-created rule limits
    When I configure custom rule limits:
      | limit_type           | value |
      | max_player_rules     | 5     |
      | max_rule_complexity  | 3     |
      | min_discovery_level  | 2     |
    Then players should be limited to 5 custom rules
    And the system should enforce complexity limits
    And players should need discovery level 2 to create rules

  Scenario: Configure rule templates
    Given a rule template configuration:
      """
      templates:
        - name: item_transformation
          pattern: "{item_a} transforms into {item_b} when {condition}"
          parameters:
            - item_a
            - item_b
            - condition
          validation_rules:
            - items_must_exist
            - condition_must_be_testable
      """
    When a player creates a rule using the template
    Then the system should validate all required parameters
    And the system should apply the template-specific validation rules
    And the created rule should follow the template structure

  Scenario: Configure rule evolution
    When I enable rule evolution with parameters:
      | parameter             | value |
      | evolution_chance      | 0.05  |
      | stability_threshold   | 0.7   |
      | adaptation_rate       | 0.1   |
    Then rules should have a 5% chance to evolve each game day
    And stable rules above 0.7 threshold should resist evolution
    And evolutions should occur at a 0.1 rate of change

  Scenario: Configure rule triggers
    When I define rule trigger types:
      | trigger_type    | description                       |
      | action          | triggered by player actions       |
      | state_change    | triggered by world state changes  |
      | time_based      | triggered by game time passage    |
      | event           | triggered by specific events      |
    Then the rules engine should monitor for appropriate triggers
    And triggered rules should be executed at the right time
    And rule execution order for simultaneous triggers should be deterministic

  Scenario: Test rule conflict resolution strategies
    Given conflicting rules for the same situation
    When I configure the conflict resolution to "priority_based"
    Then higher priority rules should take precedence
    And the system should log which rule was applied
    And the system should track override statistics

  Scenario Outline: Configure different rule systems
    When I enable the "<rule_system>" subsystem
    Then the rules engine should load "<rule_system>" specific behaviors
    And the system should apply the appropriate rule templates
    And interaction with other rule systems should be managed properly

    Examples:
      | rule_system     |
      | crafting        |
      | combat          |
      | conversation    |
      | economy         |
      | weather         |
      | magic           |

  Scenario: Rules persistence between sessions
    Given a player has created custom rules
    When the game is saved and later restored
    Then all custom rules should persist with their exact definitions
    And rule states should be preserved
    And rule execution history should be maintained

  Scenario: Configure rules monitoring and analytics
    When I enable rules analytics with parameters:
      | parameter               | value |
      | tracking_window_days    | 7     |
      | sampling_frequency      | 0.1   |
      | min_confidence_level    | 0.95  |
    Then the system should track rule applications for 7 days
    And rule effectiveness should be measured
    And underused or problematic rules should be flagged for review
