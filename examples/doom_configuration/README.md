# Doom configuration
This directory contains configuration files and doom wad files that are used by doom-rl examples.

## Scenario description

### basic.wad
Only one monster is spawned randomly at the opposite wall.  
Only one shot is needed to kill this monster.

**Agent's goal**
- **KILL** the monster with **minimum** attacks

**Episode ends**
- after the monster is killed
- on timeout

**Rewards**
- **+106** on killing the monster
- every attack: **-5**

### basic.hard.wad
One monster is spawned at a time randomly at the opposite wall. After a monster is killed, another monster 
will be spawned randomly.  
Only one shot is needed to kill the monsters.
 
**Agent's goal**
- **KILL** all the monsters with **minimum** attacks

**Episode ends**
- after **5** monsters are killed
- on timeout

**Rewards**
- **+110** on killing a monster
- every attack: **-10**

### health_gathering.wad
The floor is acid and hurts the agent periodically.  
Initially, **ten medikits** are spread randomly over the map. After picking up a medikit, another medikit 
will be generated randomly.

**Agent's goal**
- Pick up as many medikits as possible
- **STAY ALIVE**

**Episode ends**
- after player dies
- on timeout

**Rewards**
- **+120** on picking up a medikit (+20 health)

### health_gathering_hard.wad
The floor is acid and hurts the agent periodically.  
Initially, **FOUR medikits**, **three poisons**, and **eight small medikits** are spread randomly over the map. 
After picking up an item, the same item will be generated randomly.

**Agent's goal**
- Pick up as many medikits and small medikits as possible
- Avoid **poison**
- **STAY ALIVE**

**Episode ends**
- after player dies
- on timeout

**Rewards**
- **+105** on picking up a medikit (+20 health)
- **-105** on picking up a poison (-20 health)
- **+55** on picking up a small medikit (+10 health)