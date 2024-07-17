import pygame
import random

# Initialize Pygame
pygame.init()

# Set up the display
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Noita-like Sand Simulation")

# Colors
BLACK = (0, 0, 0)
SAND = (194, 178, 128)

# Create a 2D array to represent our simulation grid
grid = [[0 for _ in range(width)] for _ in range(height)]


def update_sand():
    for y in range(height - 1, 0, -1):
        for x in range(width):
            if grid[y][x] == 1:  # If this cell is sand
                # Check multiple cells below for faster falling
                for dy in range(1, 4):  # Check up to 3 cells below
                    if y + dy >= height:
                        break
                    if grid[y + dy][x] == 0:
                        # If the cell below is empty
                        grid[y][x] = 0  # Remove sand from current cell
                        grid[y + dy][x] = 1  # Move sand to cell below
                        break
                    elif x > 0 and grid[y + dy][x - 1] == 0:
                        # If the cell below and to the left is empty
                        grid[y][x] = 0
                        grid[y + dy][x - 1] = 1
                        break
                    elif x < width - 1 and grid[y + dy][x + 1] == 0:
                        # If the cell below and to the right is empty
                        grid[y][x] = 0
                        grid[y + dy][x + 1] = 1
                        break


def add_sand(x, y):
    for dx in range(-9, 10):
        for dy in range(-9, 10):
            if 0 <= x + dx < width and 0 <= y + dy < height:
                if random.random() < 0.4:
                    grid[y + dy][x + dx] = 1


def draw():
    screen.fill(BLACK)
    for y in range(height):
        for x in range(width):
            if grid[y][x] == 1:
                pygame.draw.rect(screen, SAND, (x, y, 1, 1))
    pygame.display.flip()


running = True
clock = pygame.time.Clock()
mouse_pressed = False

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pressed = True
        elif event.type == pygame.MOUSEBUTTONUP:
            mouse_pressed = False

    if mouse_pressed:
        x, y = pygame.mouse.get_pos()
        add_sand(x, y)

    update_sand()
    draw()
    clock.tick(60)

pygame.quit()
