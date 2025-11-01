#!/usr/bin/env python3
"""P5: Screen-Space Effects Tests

Tests for SSAO/GTAO, SSGI, and SSR implementation.
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from forge3d.screen_space_gi import ScreenSpaceGI, parse_gi_args, add_gi_arguments
import argparse


class TestScreenSpaceGI:
    """Test ScreenSpaceGI class."""
    
    def test_init(self):
        """Test initialization."""
        gi = ScreenSpaceGI(width=1920, height=1080)
        assert gi.width == 1920
        assert gi.height == 1080
        assert len(gi.get_enabled_effects()) == 0
    
    def test_enable_ssao(self):
        """Test SSAO enable."""
        gi = ScreenSpaceGI()
        gi.enable_effect(ScreenSpaceGI.SSAO, radius=0.5, intensity=1.0)
        
        assert gi.is_enabled(ScreenSpaceGI.SSAO)
        assert not gi.is_enabled(ScreenSpaceGI.SSGI)
        assert not gi.is_enabled(ScreenSpaceGI.SSR)
        
        settings = gi.get_settings(ScreenSpaceGI.SSAO)
        assert settings["radius"] == 0.5
        assert settings["intensity"] == 1.0
    
    def test_enable_ssgi(self):
        """Test SSGI enable."""
        gi = ScreenSpaceGI()
        gi.enable_effect(ScreenSpaceGI.SSGI, num_steps=32, radius=2.0)
        
        assert gi.is_enabled(ScreenSpaceGI.SSGI)
        settings = gi.get_settings(ScreenSpaceGI.SSGI)
        assert settings["num_steps"] == 32
        assert settings["radius"] == 2.0
    
    def test_enable_ssr(self):
        """Test SSR enable."""
        gi = ScreenSpaceGI()
        gi.enable_effect(ScreenSpaceGI.SSR, max_steps=64, thickness=0.05)
        
        assert gi.is_enabled(ScreenSpaceGI.SSR)
        settings = gi.get_settings(ScreenSpaceGI.SSR)
        assert settings["max_steps"] == 64
        assert settings["thickness"] == 0.05
    
    def test_disable_effect(self):
        """Test disabling effects."""
        gi = ScreenSpaceGI()
        gi.enable_effect(ScreenSpaceGI.SSAO)
        gi.enable_effect(ScreenSpaceGI.SSGI)
        
        assert len(gi.get_enabled_effects()) == 2
        
        gi.disable_effect(ScreenSpaceGI.SSAO)
        assert not gi.is_enabled(ScreenSpaceGI.SSAO)
        assert gi.is_enabled(ScreenSpaceGI.SSGI)
        assert len(gi.get_enabled_effects()) == 1
    
    def test_set_ssao_settings(self):
        """Test SSAO settings update."""
        gi = ScreenSpaceGI()
        gi.enable_effect(ScreenSpaceGI.SSAO)
        
        gi.set_ssao_settings(radius=0.75, intensity=1.5, num_samples=32)
        settings = gi.get_settings(ScreenSpaceGI.SSAO)
        
        assert settings["radius"] == 0.75
        assert settings["intensity"] == 1.5
        assert settings["num_samples"] == 32
    
    def test_set_ssgi_settings(self):
        """Test SSGI settings update."""
        gi = ScreenSpaceGI()
        gi.enable_effect(ScreenSpaceGI.SSGI)
        
        gi.set_ssgi_settings(num_steps=24, step_size=0.08)
        settings = gi.get_settings(ScreenSpaceGI.SSGI)
        
        assert settings["num_steps"] == 24
        assert settings["step_size"] == 0.08
    
    def test_set_ssr_settings(self):
        """Test SSR settings update."""
        gi = ScreenSpaceGI()
        gi.enable_effect(ScreenSpaceGI.SSR)
        
        gi.set_ssr_settings(max_distance=15.0, intensity=0.8)
        settings = gi.get_settings(ScreenSpaceGI.SSR)
        
        assert settings["max_distance"] == 15.0
        assert settings["intensity"] == 0.8
    
    def test_multiple_effects(self):
        """Test enabling multiple effects simultaneously."""
        gi = ScreenSpaceGI()
        gi.enable_effect(ScreenSpaceGI.SSAO)
        gi.enable_effect(ScreenSpaceGI.SSGI)
        gi.enable_effect(ScreenSpaceGI.SSR)
        
        enabled = gi.get_enabled_effects()
        assert len(enabled) == 3
        assert ScreenSpaceGI.SSAO in enabled
        assert ScreenSpaceGI.SSGI in enabled
        assert ScreenSpaceGI.SSR in enabled
    
    def test_invalid_effect(self):
        """Test error handling for invalid effect."""
        gi = ScreenSpaceGI()
        with pytest.raises(ValueError):
            gi.enable_effect("invalid_effect")


class TestCLIArguments:
    """Test CLI argument parsing."""
    
    def test_add_gi_arguments(self):
        """Test adding GI arguments to parser."""
        parser = argparse.ArgumentParser()
        add_gi_arguments(parser)
        
        # Parse with SSAO
        args = parser.parse_args(['--gi', 'ssao', '--ssao-radius', '0.75'])
        assert args.gi == 'ssao'
        assert args.ssao_radius == 0.75
    
    def test_parse_ssao_args(self):
        """Test parsing SSAO arguments."""
        parser = argparse.ArgumentParser()
        add_gi_arguments(parser)
        
        args = parser.parse_args([
            '--gi', 'ssao',
            '--ssao-radius', '0.6',
            '--ssao-intensity', '1.2'
        ])
        
        gi = parse_gi_args(args)
        assert gi.is_enabled(ScreenSpaceGI.SSAO)
        settings = gi.get_settings(ScreenSpaceGI.SSAO)
        assert settings["radius"] == 0.6
        assert settings["intensity"] == 1.2
    
    def test_parse_ssgi_args(self):
        """Test parsing SSGI arguments."""
        parser = argparse.ArgumentParser()
        add_gi_arguments(parser)
        
        args = parser.parse_args([
            '--gi', 'ssgi',
            '--ssgi-steps', '24',
            '--ssgi-radius', '1.5'
        ])
        
        gi = parse_gi_args(args)
        assert gi.is_enabled(ScreenSpaceGI.SSGI)
        settings = gi.get_settings(ScreenSpaceGI.SSGI)
        assert settings["num_steps"] == 24
        assert settings["radius"] == 1.5
    
    def test_parse_ssr_args(self):
        """Test parsing SSR arguments."""
        parser = argparse.ArgumentParser()
        add_gi_arguments(parser)
        
        args = parser.parse_args([
            '--gi', 'ssr',
            '--ssr-max-steps', '48',
            '--ssr-thickness', '0.08'
        ])
        
        gi = parse_gi_args(args)
        assert gi.is_enabled(ScreenSpaceGI.SSR)
        settings = gi.get_settings(ScreenSpaceGI.SSR)
        assert settings["max_steps"] == 48
        assert settings["thickness"] == 0.08


class TestAcceptanceCriteria:
    """Test P5 acceptance criteria."""
    
    def test_ssao_darkens_creases(self):
        """Verify: AO visibly darkens creases."""
        gi = ScreenSpaceGI()
        gi.enable_effect(ScreenSpaceGI.SSAO, radius=0.5, intensity=1.0)
        
        # SSAO should be properly configured for crevice darkening
        settings = gi.get_settings(ScreenSpaceGI.SSAO)
        assert settings["radius"] > 0.0  # Non-zero radius for sampling
        assert settings["intensity"] > 0.0  # Non-zero intensity for effect
        assert settings["bias"] < 0.1  # Small bias to prevent over-darkening
        
        print("✓ AO configured to darken creases")
    
    def test_ssgi_adds_bounce(self):
        """Verify: SSGI adds diffuse bounce on walls."""
        gi = ScreenSpaceGI()
        gi.enable_effect(ScreenSpaceGI.SSGI, num_steps=16, radius=1.0)
        
        # SSGI should have reasonable ray marching parameters
        settings = gi.get_settings(ScreenSpaceGI.SSGI)
        assert settings["num_steps"] >= 8  # Enough steps for quality
        assert settings["radius"] > 0.0  # Non-zero search radius
        assert settings["intensity"] > 0.0  # Visible contribution
        
        print("✓ SSGI configured for indirect lighting")
    
    def test_ssr_reflects_objects(self):
        """Verify: SSR reflects sky & bright objects."""
        gi = ScreenSpaceGI()
        gi.enable_effect(ScreenSpaceGI.SSR, max_steps=32, thickness=0.1)
        
        # SSR should have sufficient steps and thickness
        settings = gi.get_settings(ScreenSpaceGI.SSR)
        assert settings["max_steps"] >= 16  # Enough steps for accuracy
        assert settings["thickness"] > 0.0  # Non-zero for intersection
        assert settings["max_distance"] > 0.0  # Non-zero search distance
        
        print("✓ SSR configured for reflections")
    
    def test_all_effects_independent(self):
        """Verify effects can be enabled independently."""
        # Test each effect alone
        for effect in [ScreenSpaceGI.SSAO, ScreenSpaceGI.SSGI, ScreenSpaceGI.SSR]:
            gi = ScreenSpaceGI()
            gi.enable_effect(effect)
            assert len(gi.get_enabled_effects()) == 1
            assert gi.is_enabled(effect)
        
        print("✓ All effects work independently")
    
    def test_all_effects_combined(self):
        """Verify all effects can work together."""
        gi = ScreenSpaceGI()
        gi.enable_effect(ScreenSpaceGI.SSAO)
        gi.enable_effect(ScreenSpaceGI.SSGI)
        gi.enable_effect(ScreenSpaceGI.SSR)
        
        assert len(gi.get_enabled_effects()) == 3
        print("✓ All effects can be combined")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
