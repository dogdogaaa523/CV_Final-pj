#!/bin/bash

# 获取当前分支的名称
current_branch=$(git branch --show-current)

# 获取远程仓库的最新信息
git fetch origin

# 检查本地分支是否落后于远程分支
behind_count=$(git rev-list --count HEAD..origin/$current_branch)

if [ $behind_count -gt 0 ]; then
    echo "Your branch is behind 'origin/$current_branch' by $behind_count commits."
    echo "Pulling the latest changes from remote."
    git pull origin $current_branch
else
    echo "Your branch is up to date with 'origin/$current_branch'."
fi
