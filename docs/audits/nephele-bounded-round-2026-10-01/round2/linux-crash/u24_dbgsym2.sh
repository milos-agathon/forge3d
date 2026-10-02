#!/usr/bin/env bash
cd /root
curl -sSLf -o mvd-dbgsym.ddeb "https://launchpad.net/ubuntu/+archive/primary/+files/mesa-vulkan-drivers-dbgsym_25.2.8-0ubuntu0.24.04.3_amd64.ddeb" && dpkg -i mvd-dbgsym.ddeb && echo INSTALLED
which curl
