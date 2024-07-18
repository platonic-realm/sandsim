 
# sandsim Bash Implementation

This directory contains a Bash implementation of the sandsim project, providing a simple, terminal-based sand falling simulation.

## Overview

This Bash script creates a sand simulation that runs directly in your terminal. It adapts to your terminal size, creating a custom-fit simulation area. Sand particles fall due to gravity and interact with each other, creating interesting patterns and flows.

## Features

- Adapts to terminal size for a custom-fit simulation
- Adds small chunks of sand randomly at the top of the simulation
- Simple ASCII graphics ('#' for sand, ' ' for empty space)
- Runs entirely within a terminal, no external dependencies required

## Requirements

- Bash shell (version 4.0 or later recommended)
- A terminal that supports ANSI escape sequences for screen clearing

## Running the Simulation

1. Make sure the script is executable:
   ```
   chmod +x sandsim.sh
   ```

2. Run the script:
   ```
   ./sandsim.sh
   ```

3. The script will display the detected simulation size and prompt you to press any key to start.

4. Watch the sand fall and form patterns in your terminal.

5. To stop the simulation, press Ctrl+C.

## Customization

You can modify the following aspects of the simulation by editing the script:

- Chunk size: Adjust the `chunk_width` and `chunk_height` variables in the `add_sand_chunk` function.
- Sand addition frequency: Modify the condition in the main loop where `$((RANDOM % 5))` is used.
- Simulation speed: Change the `sleep` duration in the main loop.

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file in the root directory for details.
