#!/usr/bin/env python3
"""
Advanced Example 10: Large Texture Upload Policies

Demonstrates efficient large texture upload strategies, memory management,
and streaming policies for handling high-resolution textures within GPU memory constraints.
"""

import numpy as np
import sys
import os
import time
from pathlib import Path
from typing import Tuple, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

# Memory constraint: 512 MiB host-visible GPU memory budget
MAX_GPU_MEMORY_BYTES = 512 * 1024 * 1024

class TextureUploadPolicy:
    """Base class for texture upload policies."""
    
    def __init__(self, name: str, max_memory_bytes: int = MAX_GPU_MEMORY_BYTES):
        self.name = name
        self.max_memory_bytes = max_memory_bytes
        self.uploaded_textures = []
        self.memory_used = 0
        
    def calculate_texture_memory(self, width: int, height: int, channels: int = 4, bytes_per_channel: int = 1) -> int:
        """Calculate memory usage for a texture."""
        return width * height * channels * bytes_per_channel
    
    def can_upload(self, width: int, height: int, channels: int = 4) -> bool:
        """Check if texture can be uploaded within memory constraints."""
        required_memory = self.calculate_texture_memory(width, height, channels)
        return (self.memory_used + required_memory) <= self.max_memory_bytes
    
    def upload_texture(self, texture_data: np.ndarray, texture_id: str) -> bool:
        """Upload texture using this policy. Returns True if successful."""
        raise NotImplementedError
    
    def get_statistics(self) -> dict:
        """Get upload statistics."""
        return {
            'policy_name': self.name,
            'textures_uploaded': len(self.uploaded_textures),
            'memory_used_bytes': self.memory_used,
            'memory_used_mb': self.memory_used / (1024 * 1024),
            'memory_utilization': self.memory_used / self.max_memory_bytes,
            'average_texture_size': self.memory_used / len(self.uploaded_textures) if self.uploaded_textures else 0,
        }


class NaiveUploadPolicy(TextureUploadPolicy):
    """Naive policy: upload everything until memory runs out."""
    
    def __init__(self, max_memory_bytes: int = MAX_GPU_MEMORY_BYTES):
        super().__init__("Naive Upload", max_memory_bytes)
    
    def upload_texture(self, texture_data: np.ndarray, texture_id: str) -> bool:
        height, width = texture_data.shape[:2]
        channels = texture_data.shape[2] if len(texture_data.shape) > 2 else 1
        
        if not self.can_upload(width, height, channels):
            return False
        
        # Simulate upload
        texture_memory = self.calculate_texture_memory(width, height, channels)
        self.memory_used += texture_memory
        self.uploaded_textures.append({
            'id': texture_id,
            'size': (width, height, channels),
            'memory_bytes': texture_memory,
        })
        
        return True


class TiledUploadPolicy(TextureUploadPolicy):
    """Tiled policy: split large textures into tiles."""
    
    def __init__(self, tile_size: int = 512, max_memory_bytes: int = MAX_GPU_MEMORY_BYTES):
        super().__init__(f"Tiled Upload ({tile_size}px tiles)", max_memory_bytes)
        self.tile_size = tile_size
    
    def upload_texture(self, texture_data: np.ndarray, texture_id: str) -> bool:
        height, width = texture_data.shape[:2]
        channels = texture_data.shape[2] if len(texture_data.shape) > 2 else 1
        
        # Calculate number of tiles needed
        tiles_x = (width + self.tile_size - 1) // self.tile_size
        tiles_y = (height + self.tile_size - 1) // self.tile_size
        
        tiles_uploaded = 0
        
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                # Calculate tile bounds
                x_start = tx * self.tile_size
                y_start = ty * self.tile_size
                x_end = min(x_start + self.tile_size, width)
                y_end = min(y_start + self.tile_size, height)
                
                tile_width = x_end - x_start
                tile_height = y_end - y_start
                
                if not self.can_upload(tile_width, tile_height, channels):
                    # Return success if we uploaded at least one tile
                    return tiles_uploaded > 0
                
                # Simulate tile upload
                tile_memory = self.calculate_texture_memory(tile_width, tile_height, channels)
                self.memory_used += tile_memory
                self.uploaded_textures.append({
                    'id': f"{texture_id}_tile_{tx}_{ty}",
                    'size': (tile_width, tile_height, channels),
                    'memory_bytes': tile_memory,
                    'tile_coords': (tx, ty),
                    'parent_texture': texture_id,
                })
                tiles_uploaded += 1
        
        return tiles_uploaded > 0


class MipmapUploadPolicy(TextureUploadPolicy):
    """Mipmap policy: upload multiple resolution levels."""
    
    def __init__(self, max_mip_levels: int = 4, max_memory_bytes: int = MAX_GPU_MEMORY_BYTES):
        super().__init__(f"Mipmap Upload ({max_mip_levels} levels)", max_memory_bytes)
        self.max_mip_levels = max_mip_levels
    
    def generate_mipmap_level(self, texture: np.ndarray, level: int) -> np.ndarray:
        """Generate mipmap level by downsampling."""
        if level == 0:
            return texture
        
        # Simple box filter downsampling
        scale_factor = 2 ** level
        height, width = texture.shape[:2]
        new_height = max(1, height // scale_factor)
        new_width = max(1, width // scale_factor)
        
        if len(texture.shape) == 3:
            downsampled = np.zeros((new_height, new_width, texture.shape[2]), dtype=texture.dtype)
            for c in range(texture.shape[2]):
                downsampled[:, :, c] = self._downsample_2d(texture[:, :, c], new_height, new_width)
        else:
            downsampled = self._downsample_2d(texture, new_height, new_width)
        
        return downsampled
    
    def _downsample_2d(self, image: np.ndarray, new_height: int, new_width: int) -> np.ndarray:
        """Simple 2D downsampling using averaging."""
        old_height, old_width = image.shape
        
        # Create coordinate grids for sampling
        y_indices = np.linspace(0, old_height - 1, new_height).astype(int)
        x_indices = np.linspace(0, old_width - 1, new_width).astype(int)
        
        return image[np.ix_(y_indices, x_indices)]
    
    def upload_texture(self, texture_data: np.ndarray, texture_id: str) -> bool:
        levels_uploaded = 0
        
        for level in range(self.max_mip_levels):
            mip_texture = self.generate_mipmap_level(texture_data, level)
            height, width = mip_texture.shape[:2]
            channels = mip_texture.shape[2] if len(mip_texture.shape) > 2 else 1
            
            if not self.can_upload(width, height, channels):
                break
            
            # Upload mip level
            mip_memory = self.calculate_texture_memory(width, height, channels)
            self.memory_used += mip_memory
            self.uploaded_textures.append({
                'id': f"{texture_id}_mip_{level}",
                'size': (width, height, channels),
                'memory_bytes': mip_memory,
                'mip_level': level,
                'parent_texture': texture_id,
            })
            levels_uploaded += 1
        
        return levels_uploaded > 0


class StreamingUploadPolicy(TextureUploadPolicy):
    """Streaming policy: upload textures with LRU eviction."""
    
    def __init__(self, max_textures: int = 10, max_memory_bytes: int = MAX_GPU_MEMORY_BYTES):
        super().__init__(f"Streaming Upload (LRU, max {max_textures})", max_memory_bytes)
        self.max_textures = max_textures
        self.access_order = []  # For LRU tracking
    
    def evict_lru_texture(self):
        """Evict least recently used texture."""
        if not self.uploaded_textures or not self.access_order:
            return
        
        # Find least recently used texture
        lru_texture_id = self.access_order[0]
        lru_texture = next((t for t in self.uploaded_textures if t['id'] == lru_texture_id), None)
        
        if lru_texture:
            # Remove from memory tracking
            self.memory_used -= lru_texture['memory_bytes']
            self.uploaded_textures.remove(lru_texture)
            self.access_order.remove(lru_texture_id)
    
    def upload_texture(self, texture_data: np.ndarray, texture_id: str) -> bool:
        height, width = texture_data.shape[:2]
        channels = texture_data.shape[2] if len(texture_data.shape) > 2 else 1
        
        # Check if we need to evict textures
        while (not self.can_upload(width, height, channels) or 
               len(self.uploaded_textures) >= self.max_textures):
            if not self.uploaded_textures:
                break
            self.evict_lru_texture()
        
        if not self.can_upload(width, height, channels):
            return False
        
        # Upload texture
        texture_memory = self.calculate_texture_memory(width, height, channels)
        self.memory_used += texture_memory
        self.uploaded_textures.append({
            'id': texture_id,
            'size': (width, height, channels),
            'memory_bytes': texture_memory,
        })
        
        # Update access order
        if texture_id in self.access_order:
            self.access_order.remove(texture_id)
        self.access_order.append(texture_id)
        
        return True


def generate_test_textures() -> List[Tuple[np.ndarray, str, dict]]:
    """Generate test textures of various sizes for upload testing."""
    
    textures = []
    
    # Large high-resolution textures
    sizes_large = [(2048, 2048), (4096, 2048), (1024, 4096)]
    for i, (width, height) in enumerate(sizes_large):
        print(f"Generating large texture {i+1}: {width}x{height}...")
        
        # Create procedural texture
        x = np.linspace(0, 4*np.pi, width)
        y = np.linspace(0, 4*np.pi, height)
        X, Y = np.meshgrid(x, y)
        
        # Multi-frequency pattern
        pattern = (
            np.sin(X + i) * np.cos(Y + i * 0.5) +
            np.sin(X * 2 + i * 2) * np.cos(Y * 1.5 + i) * 0.5
        )
        
        # Convert to RGB texture
        pattern_norm = (pattern + 2) / 4  # Normalize to [0, 1]
        texture = np.zeros((height, width, 4), dtype=np.uint8)
        texture[:, :, 0] = (pattern_norm * 255).astype(np.uint8)  # R
        texture[:, :, 1] = ((1 - pattern_norm) * 255).astype(np.uint8)  # G
        texture[:, :, 2] = (np.abs(pattern_norm - 0.5) * 2 * 255).astype(np.uint8)  # B
        texture[:, :, 3] = 255  # A
        
        textures.append((texture, f"large_texture_{i}", {'type': 'large', 'size': (width, height)}))
    
    # Medium-resolution textures
    sizes_medium = [(512, 512), (1024, 512), (256, 1024)]
    for i, (width, height) in enumerate(sizes_medium):
        # Simple gradient texture
        x_grad = np.linspace(0, 1, width)
        y_grad = np.linspace(0, 1, height)
        X_grad, Y_grad = np.meshgrid(x_grad, y_grad)
        
        texture = np.zeros((height, width, 4), dtype=np.uint8)
        texture[:, :, 0] = (X_grad * 255).astype(np.uint8)
        texture[:, :, 1] = (Y_grad * 255).astype(np.uint8)
        texture[:, :, 2] = ((X_grad + Y_grad) / 2 * 255).astype(np.uint8)
        texture[:, :, 3] = 255
        
        textures.append((texture, f"medium_texture_{i}", {'type': 'medium', 'size': (width, height)}))
    
    # Small textures
    sizes_small = [(128, 128), (64, 256), (256, 64)]
    for i, (width, height) in enumerate(sizes_small):
        # Random noise texture
        np.random.seed(i + 100)
        texture = np.random.randint(0, 256, (height, width, 4), dtype=np.uint8)
        texture[:, :, 3] = 255  # Full alpha
        
        textures.append((texture, f"small_texture_{i}", {'type': 'small', 'size': (width, height)}))
    
    return textures


def test_upload_policy(policy: TextureUploadPolicy, textures: List[Tuple[np.ndarray, str, dict]]) -> dict:
    """Test a specific upload policy with the given textures."""
    
    print(f"Testing {policy.name}...")
    start_time = time.perf_counter()
    
    results = {
        'policy': policy.name,
        'textures_attempted': len(textures),
        'textures_successful': 0,
        'textures_failed': 0,
        'total_memory_requested': 0,
        'upload_results': [],
    }
    
    for texture_data, texture_id, metadata in textures:
        height, width = texture_data.shape[:2]
        channels = texture_data.shape[2] if len(texture_data.shape) > 2 else 1
        texture_memory = policy.calculate_texture_memory(width, height, channels)
        results['total_memory_requested'] += texture_memory
        
        # Attempt upload
        upload_start = time.perf_counter()
        success = policy.upload_texture(texture_data, texture_id)
        upload_time = time.perf_counter() - upload_start
        
        if success:
            results['textures_successful'] += 1
        else:
            results['textures_failed'] += 1
        
        results['upload_results'].append({
            'texture_id': texture_id,
            'success': success,
            'size': (width, height, channels),
            'memory_bytes': texture_memory,
            'upload_time': upload_time,
            'metadata': metadata,
        })
    
    end_time = time.perf_counter()
    results['total_test_time'] = end_time - start_time
    results['policy_statistics'] = policy.get_statistics()
    
    return results


def compare_upload_policies(textures: List[Tuple[np.ndarray, str, dict]]) -> dict:
    """Compare different upload policies."""
    
    print("Comparing upload policies...")
    
    # Create policies to test
    policies = [
        NaiveUploadPolicy(),
        TiledUploadPolicy(tile_size=512),
        TiledUploadPolicy(tile_size=256),
        MipmapUploadPolicy(max_mip_levels=3),
        StreamingUploadPolicy(max_textures=8),
    ]
    
    policy_results = []
    
    for policy in policies:
        result = test_upload_policy(policy, textures)
        policy_results.append(result)
    
    # Analyze comparative performance
    analysis = {
        'total_texture_memory': sum(policy.calculate_texture_memory(*tex[0].shape[:2]) for tex, _, _ in textures),
        'memory_constraint_mb': MAX_GPU_MEMORY_BYTES / (1024 * 1024),
        'policies_tested': len(policies),
        'policy_comparison': {},
    }
    
    for result in policy_results:
        policy_name = result['policy']
        stats = result['policy_statistics']
        
        analysis['policy_comparison'][policy_name] = {
            'success_rate': result['textures_successful'] / result['textures_attempted'] if result['textures_attempted'] > 0 else 0,
            'memory_efficiency': stats['memory_utilization'],
            'upload_speed': result['textures_successful'] / result['total_test_time'] if result['total_test_time'] > 0 else 0,
            'textures_uploaded': stats['textures_uploaded'],
            'average_texture_size_mb': stats['average_texture_size'] / (1024 * 1024),
        }
    
    return {
        'policy_results': policy_results,
        'comparative_analysis': analysis,
    }


def create_memory_usage_visualization(policy_results: List[dict], out_dir: Path) -> Optional[str]:
    """Create visualization of memory usage across policies."""
    
    try:
        import forge3d as f3d
        
        # Create bar chart showing memory utilization
        policy_names = [r['policy'] for r in policy_results]
        memory_usage = [r['policy_statistics']['memory_used_mb'] for r in policy_results]
        
        if not memory_usage:
            return None
        
        max_memory_mb = MAX_GPU_MEMORY_BYTES / (1024 * 1024)
        
        # Chart dimensions
        chart_width = 600
        chart_height = len(policy_names) * 60 + 100
        bar_height = 40
        
        # Create chart
        chart = np.ones((chart_height, chart_width, 4), dtype=np.uint8) * 240
        chart[:, :, 3] = 255
        
        # Color scheme
        colors = [
            (220, 100, 100),  # Red
            (100, 220, 100),  # Green
            (100, 100, 220),  # Blue  
            (220, 220, 100),  # Yellow
            (220, 100, 220),  # Magenta
        ]
        
        # Draw memory constraint line
        constraint_x = int((max_memory_mb / max_memory_mb) * (chart_width - 150)) + 100
        chart[:, constraint_x:constraint_x+2, :3] = (255, 0, 0)  # Red line
        
        # Draw bars
        for i, (policy_name, usage_mb) in enumerate(zip(policy_names, memory_usage)):
            y_pos = i * 60 + 50
            bar_width = int((usage_mb / max_memory_mb) * (chart_width - 150))
            color = colors[i % len(colors)]
            
            # Draw bar
            if bar_width > 0:
                chart[y_pos:y_pos+bar_height, 100:100+bar_width, :3] = color
        
        # Save chart
        chart_path = out_dir / "texture_upload_memory_usage.png"
        f3d.numpy_to_png(str(chart_path), chart)
        
        return str(chart_path)
        
    except Exception as e:
        print(f"Memory usage visualization failed: {e}")
        return None


def main():
    """Main example execution."""
    print("Large Texture Upload Policies")
    print("=============================")
    
    out_dir = Path(__file__).parent.parent / "out"
    out_dir.mkdir(exist_ok=True)
    
    try:
        import forge3d as f3d
        
        # Generate test textures
        print("Generating test textures...")
        textures = generate_test_textures()
        
        # Calculate total texture memory
        total_memory = sum(
            len(tex[0].data) for tex, _, _ in textures
        )
        print(f"Generated {len(textures)} textures, total size: {total_memory / (1024*1024):.1f} MB")
        
        # Test all upload policies
        comparison_results = compare_upload_policies(textures)
        
        # Create visualizations
        print("Creating visualizations...")
        
        # Save a sample of successful uploads
        saved_paths = {}
        sample_count = 0
        
        for policy_result in comparison_results['policy_results']:
            policy_name_safe = policy_result['policy'].replace(' ', '_').replace('(', '').replace(')', '')
            
            for upload_result in policy_result['upload_results'][:3]:  # Save first 3
                if upload_result['success'] and sample_count < 12:  # Limit total samples
                    # Find corresponding texture
                    texture_data = next(
                        (tex[0] for tex, tex_id, _ in textures if tex_id == upload_result['texture_id']),
                        None
                    )
                    
                    if texture_data is not None:
                        # Resize if too large for demo output
                        if texture_data.shape[0] > 512 or texture_data.shape[1] > 512:
                            # Simple downsampling
                            scale = min(512 / texture_data.shape[0], 512 / texture_data.shape[1])
                            new_height = int(texture_data.shape[0] * scale)
                            new_width = int(texture_data.shape[1] * scale)
                            
                            # Simple nearest neighbor downsampling
                            indices_y = np.linspace(0, texture_data.shape[0] - 1, new_height).astype(int)
                            indices_x = np.linspace(0, texture_data.shape[1] - 1, new_width).astype(int)
                            texture_data = texture_data[np.ix_(indices_y, indices_x)]
                        
                        sample_path = out_dir / f"texture_upload_{policy_name_safe}_{upload_result['texture_id']}.png"
                        f3d.numpy_to_png(str(sample_path), texture_data)
                        
                        if policy_name_safe not in saved_paths:
                            saved_paths[policy_name_safe] = []
                        saved_paths[policy_name_safe].append(str(sample_path))
                        sample_count += 1
        
        # Create memory usage visualization
        memory_chart_path = create_memory_usage_visualization(comparison_results['policy_results'], out_dir)
        if memory_chart_path:
            saved_paths['memory_usage_chart'] = memory_chart_path
        
        # Generate comprehensive metrics
        metrics = {
            'test_configuration': {
                'textures_tested': len(textures),
                'memory_constraint_mb': MAX_GPU_MEMORY_BYTES / (1024 * 1024),
                'total_texture_memory_mb': total_memory / (1024 * 1024),
                'policies_tested': len(comparison_results['policy_results']),
            },
            'texture_breakdown': {},
            'policy_performance': {},
            'upload_strategies': [
                'naive_until_full',
                'tiled_decomposition', 
                'mipmap_levels',
                'lru_streaming',
                'memory_constraint_enforcement'
            ],
            'outputs': saved_paths,
        }
        
        # Analyze texture breakdown by type
        texture_types = {}
        for texture_data, texture_id, metadata in textures:
            tex_type = metadata.get('type', 'unknown')
            if tex_type not in texture_types:
                texture_types[tex_type] = {'count': 0, 'total_memory_mb': 0}
            texture_types[tex_type]['count'] += 1
            texture_types[tex_type]['total_memory_mb'] += len(texture_data.data) / (1024 * 1024)
        
        metrics['texture_breakdown'] = texture_types
        
        # Add policy performance metrics
        for policy_result in comparison_results['policy_results']:
            policy_name = policy_result['policy']
            stats = policy_result['policy_statistics']
            
            metrics['policy_performance'][policy_name] = {
                'success_rate': policy_result['textures_successful'] / policy_result['textures_attempted'],
                'memory_utilization': stats['memory_utilization'],
                'textures_uploaded': stats['textures_uploaded'],
                'memory_used_mb': stats['memory_used_mb'],
                'average_texture_size_mb': stats['average_texture_size'] / (1024 * 1024),
                'test_time_seconds': policy_result['total_test_time'],
            }
        
        # Print performance summary
        print("\nTexture Upload Policy Results:")
        print(f"{'Policy':<30} {'Success':<8} {'Memory':<10} {'Textures':<10} {'Efficiency':<12}")
        print("-" * 80)
        
        for policy_name, perf in metrics['policy_performance'].items():
            print(f"{policy_name:<30} {perf['success_rate']:<8.1%} "
                  f"{perf['memory_used_mb']:<10.1f} {perf['textures_uploaded']:<10} "
                  f"{perf['memory_utilization']:<12.1%}")
        
        print(f"\nMemory constraint: {metrics['test_configuration']['memory_constraint_mb']:.0f} MB")
        print(f"Total texture data: {metrics['test_configuration']['total_texture_memory_mb']:.1f} MB")
        print(f"Constraint utilization: {(metrics['test_configuration']['memory_constraint_mb'] / metrics['test_configuration']['total_texture_memory_mb']):.1%}")
        
        # Save metrics
        import json
        metrics_path = out_dir / "texture_upload_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics: {metrics_path}")
        
        print("\nExample completed successfully!")
        return 0
        
    except ImportError as e:
        print(f"forge3d not available: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())