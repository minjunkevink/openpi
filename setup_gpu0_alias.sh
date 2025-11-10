#!/bin/bash
# Setup script to add gpu0 alias to your shell
# Run this script with: source setup_gpu0_alias.sh

# Function to add alias to shell config
add_alias_to_shell() {
    local shell_config=$1
    local alias_line='alias gpu0="export CUDA_VISIBLE_DEVICES=0"'
    
    if [ -f "$shell_config" ]; then
        if ! grep -q "alias gpu0" "$shell_config"; then
            echo "" >> "$shell_config"
            echo "# OpenPI GPU0 alias" >> "$shell_config"
            echo "$alias_line" >> "$shell_config"
            echo "Added gpu0 alias to $shell_config"
        else
            echo "gpu0 alias already exists in $shell_config"
        fi
    fi
}

# Detect shell and add alias
if [ -n "$ZSH_VERSION" ]; then
    # Zsh
    add_alias_to_shell "$HOME/.zshrc"
    echo "To use the alias now, run: source ~/.zshrc"
elif [ -n "$BASH_VERSION" ]; then
    # Bash
    add_alias_to_shell "$HOME/.bashrc"
    echo "To use the alias now, run: source ~/.bashrc"
else
    echo "Shell not detected. Please manually add this alias to your shell config:"
    echo 'alias gpu0="export CUDA_VISIBLE_DEVICES=0"'
fi

# Export the alias for current session
export CUDA_VISIBLE_DEVICES=0
alias gpu0="export CUDA_VISIBLE_DEVICES=0"
echo ""
echo "gpu0 alias is now active in this session!"
echo "You can use it by typing: gpu0"

