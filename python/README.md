# sandsim Python Implementation

This directory contains a Python implementation of the sandsim project, using Pygame for visualization of the sand falling simulation.

## Requirements

- Python 3.6+
- Pygame

To install Pygame, you can use pip:

```
pip install pygame
```

## Running the Simulation

To run the simulation, execute the Python script:

```
python sandsim.py
```

## Implementation Details

The `sandsim.py` file contains all the logic for the sand simulation and visualization. It uses Pygame to render the simulation in a window.

Key components:
- Grid representation using a 2D list
- Pygame for rendering the simulation state
- Basic Python operations to update sand positions

Key functions:
- `initialize_grid()`: Sets up the initial sand distribution.
- `update_sand()`: Performs one step of the sand falling simulation.
- `draw_grid()`: Renders the current state of the grid using Pygame.
- `main()`: Runs the simulation loop and handles Pygame events.

## Customization

You can modify the following variables at the top of the script to customize the simulation:

- `WIDTH`: The width of the simulation grid and window.
- `HEIGHT`: The height of the simulation grid and window.
- `CELL_SIZE`: The size of each cell in pixels.
- `FPS`: Frames per second for the simulation update and rendering.

## Controls

- Left mouse button: Add sand at the cursor position
- Right mouse button: Remove sand at the cursor position
- Close the window to exit the simulation

## Performance Considerations

This implementation uses basic Python operations for the simulation logic. While Pygame provides efficient rendering, the core simulation might be slower compared to optimized or compiled implementations, especially for larger grids.

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file in the root directory for details.