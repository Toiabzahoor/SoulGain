#!/bin/bash

echo "🌸 Starting Infinite SoulGain Evolution Loop 🌸"
echo "Press Ctrl+C at any time to safely stop. (Your atomic saves will protect the memory!)"

while true; do
    echo "=================================================="
    echo "🌱 Birthing a new batch of 20 games..."
    echo "=================================================="
    
    # We use printf to automatically send '3' (for the menu) and '20' (for the games)
    printf "3\n20\n" | cargo run --bin soulgain --release
    
    echo "✨ Batch complete! Taking a quick breath before the next round..."
    sleep 2 # A tiny 2-second pause so you can easily Ctrl+C between batches if you want!
done