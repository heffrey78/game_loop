# Game Loop Player Guide

Welcome to Game Loop, an immersive text adventure game powered by natural language processing and dynamic world generation. This guide will help you navigate and enjoy your adventure.

## Table of Contents
- [Getting Started](#getting-started)
- [Basic Commands](#basic-commands)
- [Movement and Exploration](#movement-and-exploration)
- [Inventory Management](#inventory-management)
- [Interacting with NPCs](#interacting-with-npcs)
- [Examining Your World](#examining-your-world)
- [Game Management](#game-management)
- [Advanced Features](#advanced-features)
- [Tips and Tricks](#tips-and-tricks)
- [Troubleshooting](#troubleshooting)

## Getting Started

### Starting the Game
```bash
make run
# or
poetry run python -m game_loop.main
```

### Your First Steps
When you start the game, you'll find yourself in a location with a description of your surroundings. The game uses natural language processing, so you can interact using normal English phrases.

Try these commands to get oriented:
- `look` - See your current surroundings
- `inventory` - Check what you're carrying
- `help` - Get assistance with commands

## Basic Commands

Game Loop understands natural language, so you can phrase commands in multiple ways:

### Command Variations
Most commands have several ways to express them:
- **Movement**: `north`, `n`, `go north`, `walk north`
- **Looking**: `look`, `l`, `look around`, `examine surroundings`
- **Inventory**: `inventory`, `i`, `items`, `check inventory`

### Command Structure
Commands typically follow these patterns:
- Simple commands: `look`, `inventory`, `help`
- Commands with objects: `take sword`, `examine door`, `use key`
- Commands with targets: `talk to merchant`, `use key on door`

## Movement and Exploration

### Basic Movement
Move between locations using directional commands:
- `north` or `n` - Move north
- `south` or `s` - Move south
- `east` or `e` - Move east
- `west` or `w` - Move west
- `up` or `u` - Move up (stairs, ladders)
- `down` or `d` - Move down

### Examples
```
> north
You head north.

Abandoned Factory Entrance
A rusty metal door stands before you, its paint peeling and hinges creaking. 
The industrial complex stretches out in multiple directions.

There is an exit to the south.
You see: rusty key, metal pipe.
Security Guard is here.
```

### Dynamic World Generation
Game Loop creates new locations as you explore:
- The world expands based on your exploration patterns
- Each area maintains thematic consistency
- Your preferences influence what types of locations are generated
- Hidden areas and secret passages may be discovered

### Navigation Tips
- Use `look` to see available exits
- Pay attention to location descriptions for clues about interesting directions
- Some exits may be hidden or require specific actions to discover
- The game remembers your exploration patterns and adapts accordingly

## Inventory Management

### Viewing Your Inventory
```
> inventory
Inventory:
  • Rusty Key
  • Metal Pipe (worn)
  • Health Potion (x2)

Carrying: 4/10 items, 8.5/100.0 weight
```

### Taking Items
Pick up items from your surroundings:
```
> take sword
You take the ancient sword.

> get all
You take everything you can carry.

> pick up the glowing crystal
You take the glowing crystal.
```

### Dropping Items
Remove items from your inventory:
```
> drop sword
You drop the ancient sword.

> discard empty bottle
You drop the empty bottle.

> put down heavy armor
You drop the heavy armor.
```

### Inventory Limits
- **Item Count**: You can carry a limited number of items (usually 10)
- **Weight**: Items have weight that affects what you can carry
- **Special Items**: Some quest items cannot be dropped

## Interacting with NPCs

### Starting Conversations
Talk to non-player characters you encounter:
```
> talk to merchant
Merchant looks at you. "Welcome to my shop! Are you interested in trade?"

> speak with guard
Guard stands at attention and speaks formally. "State your business here."

> chat with scholar
Scholar adjusts their glasses and speaks knowledgeably. "Ah, a fellow seeker of knowledge!"
```

### Conversation Features
- **Dynamic Dialogue**: NPCs respond based on their personality and your relationship
- **Relationship Tracking**: NPCs remember your interactions
- **Topic-Specific Responses**: Ask about specific subjects
- **Mood and Context**: NPC responses vary based on their current state

### Advanced Conversation
```
> talk to wizard about magic
Wizard mentions they might have something for you to do.

> ask merchant about prices
Merchant smiles and discusses current market rates.
```

## Examining Your World

### General Observation
Get detailed information about your surroundings:
```
> look
Industrial Workshop
A cluttered workspace filled with mechanical devices and half-finished contraptions. 
Workbenches line the walls, covered in tools and blueprints.

There are exits to the north and east.
You see: wrench, blueprint, oil lamp.
Inventor is here.
```

### Examining Specific Objects
Look closely at items, NPCs, or features:
```
> examine sword
Ancient Sword
A well-crafted blade with intricate engravings along its length. 
Despite its age, the edge remains remarkably sharp.

> inspect door
The heavy wooden door is reinforced with iron bands. 
You notice a small keyhole near the handle.

> look at merchant
A middle-aged trader with weathered hands and keen eyes. 
They seem to be in good spirits.
```

### What You Can Examine
- **Items**: Both in your inventory and in locations
- **NPCs**: Get descriptions of characters you meet
- **Environmental Features**: Doors, walls, furniture, exits
- **Hidden Details**: Some examinations reveal secrets

## Game Management

### Saving Your Progress
```
> save
Game saved successfully.

> save my_adventure
Game saved as 'my_adventure'.
```

### Loading Games
```
> load
Loading most recent save...

> load my_adventure
Loading 'my_adventure'...

> list saves
Available saves:
  • autosave_2024_01_15
  • my_adventure
  • factory_exploration
```

### Getting Help
```
> help
General Commands:
  • Movement: Use directions like 'north', 'south', 'east', 'west'
  • Inventory: Use 'inventory' to see items, 'take' and 'drop' to manage them
  • Interaction: Use 'examine' to look at things, 'talk to' for conversations
  • System: Use 'save' and 'load' for progress, 'help' for assistance

> help movement
Movement: Use 'go <direction>' or just '<direction>' to move. 
Valid directions: north, south, east, west, up, down.
```

### Exiting the Game
```
> quit
Game saved automatically. Thanks for playing!

> exit
Don't forget to save your progress! Thanks for playing!
```

## Advanced Features

### Using Items
Interact with objects in various ways:
```
> use key
You use the rusty key.

> use key on door
You unlock the door with the rusty key.

> use potion
You drink the healing potion and feel refreshed.
```

### Complex Interactions
The game supports sophisticated interactions:
```
> put gem in container
You place the glowing gem in the ornate container.

> combine wire with battery
You connect the wire to the battery, creating a simple circuit.
```

### Rules and Game Logic
Game Loop includes an intelligent rules engine that:
- **Prevents Invalid Actions**: Stops you from breaking game logic
- **Provides Guidance**: Suggests alternatives when actions fail
- **Tracks Progress**: Monitors achievements and story progression
- **Adapts Difficulty**: Adjusts challenges based on your experience

### Natural Language Processing
The game understands complex natural language:
```
> carefully examine the mysterious glowing artifact on the pedestal
> talk to the old wizard about ancient magic and forgotten spells
> pick up the sword and shield, but leave the heavy armor
```

## Tips and Tricks

### Exploration Strategies
- **Be Thorough**: Examine everything - hidden details reward careful observation
- **Try Different Directions**: Not all exits are immediately obvious
- **Follow Your Interests**: The game adapts to your exploration preferences
- **Experiment**: Try unusual commands and combinations

### Communication Tips
- **Be Natural**: Use normal English phrases
- **Be Specific**: "examine the red book" vs "examine book"
- **Try Variations**: If one phrasing doesn't work, try another
- **Use Context**: The game understands what you're referring to

### Inventory Management
- **Organize Regularly**: Keep only what you need
- **Plan Ahead**: Consider weight and space limitations
- **Check Conditions**: Items can wear out or break
- **Remember Limits**: You can't carry everything

### NPC Interactions
- **Build Relationships**: Regular interaction improves NPC disposition
- **Ask Questions**: NPCs often have valuable information
- **Be Patient**: Some information is revealed over time
- **Pay Attention**: NPC moods and responses provide clues

### Problem Solving
- **Read Carefully**: Location descriptions contain important clues
- **Think Creatively**: Multiple solutions often exist
- **Use Your Inventory**: Items you carry might solve puzzles
- **Talk to NPCs**: They often provide hints and guidance

## Troubleshooting

### Common Issues

**"I don't understand that command"**
- Try rephrasing using simpler words
- Break complex actions into smaller steps
- Use `help` for command suggestions

**"You can't do that here"**
- Check if you're in the right location
- Make sure you have required items
- Some actions require specific conditions

**"You don't see that here"**
- Use `look` to see what's available
- Try different names for objects
- Some items might be hidden

**Movement Problems**
- Use `look` to see available exits
- Check if paths are blocked
- Some areas require specific items or actions to access

**Inventory Issues**
- Check if your inventory is full
- Verify you have the items you think you do
- Some items can't be dropped (quest items)

### Getting Unstuck
If you're not sure what to do:
1. Use `look` to re-examine your surroundings
2. Check your `inventory` for useful items
3. Try `examine` on objects you haven't looked at closely
4. `talk to` any NPCs present
5. Use `help` for command reminders

### Performance Tips
- Save regularly to preserve progress
- Use specific object names when possible
- Clear commands are processed faster than ambiguous ones

## Advanced Gameplay

### Discovery and Secrets
- **Hidden Objects**: Use `examine` thoroughly to find concealed items
- **Secret Passages**: Some locations have hidden exits
- **Environmental Puzzles**: Use items and observations to solve challenges
- **NPC Secrets**: Build relationships to unlock special information

### Adaptive Content
Game Loop learns from your play style:
- **Preferred Locations**: The game generates more of what you explore
- **Difficulty Scaling**: Challenges adapt to your skill level
- **Content Variety**: Your choices influence what you encounter
- **Personal Narrative**: The story adapts to your decisions

### Meta-Gaming Features
- **Statistics Tracking**: The game monitors your preferences and progress
- **Quality Assessment**: Content generation improves based on your interactions
- **Behavior Analysis**: Your play style influences future content

## Conclusion

Game Loop offers a rich, adaptive gaming experience that evolves with your play style. The combination of natural language processing, dynamic world generation, and intelligent NPCs creates a unique adventure every time you play.

Remember:
- **Experiment freely** - the game is designed to understand natural language
- **Explore thoroughly** - new content is generated as you discover the world
- **Engage with NPCs** - they're intelligent and remember your interactions
- **Save often** - preserve your progress and achievements

Most importantly, have fun! The game is designed to be engaging and responsive to your creativity and curiosity.

---

*For technical issues or questions not covered in this guide, check the game's help system with the `help` command or refer to the project documentation.*