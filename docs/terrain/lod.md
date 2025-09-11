---
title: Terrain LOD (Impostors & Morphing)
---

# Terrain LOD System

This page outlines the intended LOD system with continuous transitions and impostor support. The implementation is staged and partially scaffolded in `src/terrain/lod.rs` and `src/terrain/impostors.rs`.

## Goals
- Reduce triangles by 50–90% in typical scenes via LOD selection.
- Maintain update budget under 16 ms during scripted sweeps.
- Avoid visible popping using morphing/impostor transitions.

## Components
- Screen-space error calculation and per-tile LOD selection.
- Impostor atlas generation and sampling (WGSL scaffold in `shaders/impostor_atlas.wgsl`).
- Integration hooks for streaming and memory budgeting.

## Status
- LOD selection helpers exist; impostor atlas and demos/tests to be completed.

