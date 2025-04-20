Feature: Game Loop Main Flow
  As a player in a text-based adventure game
  I want to interact with the game world using natural language
  So that I can have an immersive and responsive gaming experience

  Background:
    Given the game has been initialized
    And the player has started a new game
    And the player is in the "Forgotten Library" location

  # Basic Navigation Scenarios
  Scenario: Player moves to a connected location
    Given the location "Ancient Study" is connected to the current location
    When the player types "go to the ancient study"
    Then the game should process the movement command
    And the player's location should change to "Ancient Study"
    And the game should display the description of "Ancient Study"
    And the game should save the new player state

  Scenario: Player attempts to move to an invalid location
    Given the location "Secret Laboratory" is not connected to the current location
    When the player types "go to the secret laboratory"
    Then the game should indicate the location is not accessible
    And the player's location should remain unchanged

  # Object Inspection Scenarios
  Scenario: Player inspects an object in the location
    Given the current location contains an object "ancient book"
    When the player types "examine the ancient book"
    Then the game should provide a detailed description of the "ancient book"
    And the game should update player's knowledge about the object

  Scenario: Player inspects a hidden object
    Given the current location contains a hidden object "secret lever"
    When the player types "look behind the bookshelf"
    Then the game should reveal the "secret lever"
    And the game should provide a description of the "secret lever"
    And the game should update the location state to mark the object as discovered

  # Object Interaction Scenarios
  Scenario: Player takes an object
    Given the current location contains a takeable object "brass key"
    When the player types "take the brass key"
    Then the game should add the "brass key" to the player's inventory
    And the game should remove the "brass key" from the current location
    And the game should acknowledge the player has taken the "brass key"

  Scenario: Player uses an object from inventory
    Given the player has "brass key" in their inventory
    And the current location contains a "locked chest" that requires the "brass key"
    When the player types "unlock the chest with the brass key"
    Then the game should update the state of the "locked chest" to "unlocked chest"
    And the game should describe the result of the action
    And the game should reveal any contents of the chest

  Scenario: Player combines objects
    Given the player has "empty flask" in their inventory
    And the player has "healing herbs" in their inventory
    And the recipe for "healing potion" is an "empty flask" and "healing herbs"
    When the player types "combine flask with herbs"
    Then the game should remove both items from inventory
    And the game should add "healing potion" to the player's inventory
    And the game should describe the crafting process

  # Conversation Scenarios
  Scenario: Player talks to an NPC
    Given the current location contains an NPC "Old Librarian"
    When the player types "talk to the librarian"
    Then the game should query relevant context about the "Old Librarian"
    And the game should display a greeting from the "Old Librarian" appropriate to the retrieved context

  Scenario: Player asks NPC about an existing topic
    Given the player is in conversation with "Old Librarian"
    When the player types "ask about the ancient artifact"
    Then the game should retrieve relevant information about "ancient artifact"
    And the game should display the NPC's response based on their knowledge
    And the game should update the player's knowledge state  
    
  Scenario: Player asks NPC about a new topic
    Given the player is in conversation with "Old Librarian"
    When the player types "ask about the ancient artifact"
    And the game has no relevant information about the "ancient artifact"
    Then the game should decide whether to populate the world with the "ancient artifact"
    And the game should display the NPC's response based on their knowledge
    And the game should update the player's knowledge state

  # Dynamic World Discovery Scenarios
  Scenario: Player discovers an undefined location
    Given the player is in the "Forgotten Library" location
    And there is a door labeled "Mysterious Door" leading to an undefined location
    When the player types "open mysterious door"
    Then the game should generate a new location on-the-fly
    And the game should describe the newly generated location
    And the game should track this new location in the world model
    And the player's location should change to the new location

  Scenario: Player returns to a dynamically generated location
    Given the player has previously discovered a dynamically generated location "Strange Garden"
    When the player types "go to strange garden"
    Then the game should load the previously generated "Strange Garden" location
    And the game should display the description of "Strange Garden"
    And the player's location should change to "Strange Garden"

  Scenario: Player discovers a new region through exploration
    Given the player is in the "Eastern Corridor" location
    When the player types "climb through the broken window"
    And "broken window" is not a pre-defined exit
    Then the game should determine an appropriate new region to generate
    And the game should generate a new exterior location
    And the game should connect the new location to "Eastern Corridor" via the "broken window"
    And the player's location should change to the new location

  Scenario: Player uncovers a secret location through environmental interaction
    Given the current location contains an interactive object "strange bookshelf"
    When the player types "pull on the strange bookshelf"
    Then the game could generate a secret location "Hidden Study"
    And the game would describe the bookshelf moving to reveal a passage
    And the game should provide the option to enter the new location

  # World Building Scenarios
  Scenario: Player influences dynamically generated content through actions
    Given the player has performed actions suggesting interest in "ancient magic"
    When the player discovers a new location
    Then the game should tailor the generated location to include "ancient magic" elements
    And the description should reference themes consistent with player's past choices

  Scenario: Player triggers generation of NPCs in a new location
    Given the player has entered a dynamically generated location "Trade District"
    When the player types "look for people to talk to"
    Then the game should generate appropriate NPCs for the "Trade District" location
    And the game should describe the NPCs present
    And the game should enable interaction with these newly generated NPCs

  Scenario: Player triggers a dynamic quest opportunity
    Given the player is exploring a dynamically generated location
    When the player investigates a distinctive feature of the location
    Then the game should generate a quest opportunity related to the location
    And the game should present this opportunity to the player
    And the game should track the player's response to the opportunity

  # Advanced Discovery Mechanics
  Scenario: Player attempts to create a passage
    Given the player has an item "magical chalk" in their inventory
    When the player types "draw a doorway on the wall with magical chalk"
    Then the game should generate a new doorway in the current location
    And the game should create a new location accessible through this doorway
    And the game should describe the creation process and the new doorway

  Scenario Outline: Player explores beyond defined world boundaries
    Given the player is at a location on the edge of the defined world "<edge_location>"
    When the player attempts to travel "<direction>"
    Then the game should generate a new region beyond the current boundaries
    And the game should connect this region logically to the existing world
    And the player should be able to explore the new region

    Examples:
      | edge_location     | direction |
      | Northern Cliffs   | north     |
      | Eastern Wastelands| east      |
      | Southern Harbor   | south     |
      | Western Forest    | west      |

  Scenario: Player's actions trigger world evolution
    Given the player has made significant changes to a location
    When a game day passes
    Then the game should evolve the affected locations based on player actions
    And NPCs should respond to the changes in their environment
    And new opportunities should emerge from the altered state

  Scenario: Player discovers a nexus point for rapid travel
    Given the player has explored multiple regions of the world
    When the player discovers a "magical nexus" location
    Then the game should generate connections to previously visited key locations
    And the player should be able to use these connections for rapid travel
    And the nexus should adapt to include new significant locations as they're discovered

  # World Persistence and Continuity
  Scenario: World state persists between play sessions
    Given the player has discovered several dynamically generated locations
    When the player saves and exits the game
    And the player later loads the saved game
    Then all dynamically generated locations should be preserved
    And connections between locations should remain intact
    And the player should be able to navigate the entire discovered world

  # Error Recovery for Dynamic World
  Scenario: Game recovers from failed world generation
    Given the system encounters an error while generating a new location
    When the player attempts to access this location
    Then the game should gracefully handle the error
    And the game should provide a fallback location
    And the player should receive an in-world explanation for the unexpected environment

  # Game State Management Scenarios
  Scenario: Player saves the game
    When the player types "save game"
    Then the game should store the current game state
    And the game should confirm the save was successful
    And the game should continue running

  Scenario: Player loads a saved game
    Given a saved game exists
    When the player types "load game"
    Then the game should restore the saved game state
    And the game should display the current location description
    And the player should have the same inventory as in the saved game

  # Advanced Interaction Scenarios
  Scenario: Player solves a puzzle
    Given the current location contains a puzzle "star pattern lock"
    And the player has discovered a clue about "celestial alignment"
    When the player types "arrange stars according to celestial alignment"
    Then the game should recognize the puzzle solution
    And the game should update the puzzle state to "solved"
    And the game should reveal the puzzle reward
    And the game should update the player's progress

  Scenario Outline: Player performs different actions with an object
    Given the current location contains an object "<object>"
    When the player types "<action> <object>"
    Then the game should process the command 
    And the game should respond with appropriate feedback for "<action>" on "<object>"

    Examples:
      | object          | action    |
      | dusty book      | open      |
      | ancient scroll  | read      |
      | crystal orb     | touch     |
      | wooden box      | break     |
      | wall inscription| translate |

  # Error Handling Scenarios
  Scenario: Player enters an ambiguous command
    When the player types "use key"
    And there are multiple objects that could be used with a key
    Then the game should ask for clarification
    And the game should provide options for the player to choose from

  Scenario: Player enters an unrecognized command
    When the player types "dance around wildly"
    And the command does not match any known patterns
    Then the game should respond with an appropriate message

  # Complex Game Flow Example
  Scenario: Complete quest sequence
    Given the player has accepted the quest "Forbidden Knowledge"
    And the quest requires finding "Ancient Manuscript" in the "Hidden Archives"
    
    # First part - finding the location
    When the player types "ask librarian about hidden archives"
    Then the game should update the player's knowledge with location of "Hidden Archives"
    
    When the player types "go to eastern corridor"
    Then the player's location should change to "Eastern Corridor"
    
    When the player types "search for hidden door"
    Then the game should reveal "Concealed Entrance"
    
    When the player types "use concealed entrance"
    Then the player's location should change to "Hidden Archives"
    
    # Second part - puzzle solving to get the item
    When the player types "examine locked cabinet"
    Then the game should describe "An ornate cabinet with a complex mechanical lock"
    
    When the player types "examine mechanical lock"
    Then the game should describe "The lock has three rotating rings with symbols"
    
    When the player types "rotate rings to match the constellation pattern"
    Then the game should update the cabinet state to "unlocked"
    
    When the player types "open cabinet"
    Then the game should reveal "Ancient Manuscript"
    
    When the player types "take ancient manuscript"
    Then the game should add "Ancient Manuscript" to player's inventory
    
    # Third part - returning and completing the quest
    When the player types "return to forgotten library"
    Then the player's location should change to "Forgotten Library"
    
    When the player types "give manuscript to librarian"
    Then the game should remove "Ancient Manuscript" from inventory
    And the game should mark quest "Forbidden Knowledge" as completed
    And the game should reward the player with "Arcane Knowledge" skill
    And the game should advance the main storyline

  # Dynamic Rules System Scenario
  Scenario: Player creates a new game rule
    Given the game has dynamic rules system enabled
    When the player types "create rule: glowing objects can be used as light sources"
    Then the game should validate rule consistency
    And the game should add the new rule to the world model
    And the game should confirm the rule creation
    
    When the player later types "use crystal orb to light the dark room"
    And the "crystal orb" has the property "glowing"
    Then the game should apply the custom rule
    And the game should change the room property from "dark" to "illuminated"