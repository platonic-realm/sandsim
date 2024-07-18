#!/bin/bash

# Detect terminal size
read -r LINES COLUMNS < <(stty size)

# Set up the grid dimensions (leaving some margin)
WIDTH=$((COLUMNS))
HEIGHT=$((LINES - 1))

# Initialize the grid
declare -A grid
for ((y=0; y<HEIGHT; y++)); do
    for ((x=0; x<WIDTH; x++)); do
        grid[$y,$x]=" "
    done
done

# Function to update the sand
update_sand() {
    for ((y=HEIGHT-2; y>=0; y--)); do
        for ((x=0; x<WIDTH; x++)); do
            if [[ ${grid[$y,$x]} == "#" ]]; then
                if [[ ${grid[$((y+1)),$x]} == " " ]]; then
                    grid[$y,$x]=" "
                    grid[$((y+1)),$x]="#"
                elif [[ $x -gt 0 && ${grid[$((y+1)),$((x-1))]} == " " ]]; then
                    grid[$y,$x]=" "
                    grid[$((y+1)),$((x-1))]="#"
                elif [[ $x -lt $((WIDTH-1)) && ${grid[$((y+1)),$((x+1))]} == " " ]]; then
                    grid[$y,$x]=" "
                    grid[$((y+1)),$((x+1))]="#"
                fi
            fi
        done
    done
}

# Function to draw the grid
draw_grid() {
    clear
    for ((y=0; y<HEIGHT; y++)); do
        for ((x=0; x<WIDTH; x++)); do
            echo -n "${grid[$y,$x]}"
        done
        echo
    done
}

# Function to add a chunk of sand
add_sand() {
    local center_x=$1
    local chunk_width=5
    local chunk_height=3

    for ((y=0; y<chunk_height; y++)); do
        for ((x=center_x-chunk_width/2; x<center_x+chunk_width/2; x++)); do
            if [[ $x -ge 0 && $x -lt WIDTH && $y -lt HEIGHT ]]; then
                if [[ $((RANDOM % 2)) -eq 0 ]]; then
                    grid[$y,$x]="#"
                fi
            fi
        done
    done
}

# Display initial information
echo "Terminal size: ${WIDTH}x${HEIGHT}"
echo "Press any key to start the simulation..."
read -n 1 -s

# Main loop
while true; do
    # Add a chunk of sand randomly at the top
    if [[ $((RANDOM % 5)) -eq 0 ]]; then
        rand_x=$((RANDOM % WIDTH))
        add_sand $rand_x
    fi

    update_sand
    draw_grid
    sleep 0.1
done
