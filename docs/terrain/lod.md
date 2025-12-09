---
title: Terrain LOD (Impostors & Morphing)
---

# Terrain LOD System

This page outlines the intended LOD system with continuous transitions. The implementation is staged in `src/terrain/lod.rs`.

## Goals
- Reduce triangles by 50–90% in typical scenes via LOD selection.
- Maintain update budget under 16 ms during scripted sweeps.
- Avoid visible popping using morphing transitions.

## Components
- Screen-space error calculation and per-tile LOD selection.
- Integration hooks for streaming and memory budgeting.

## Status
- LOD selection helpers exist; demos/tests to be completed.

