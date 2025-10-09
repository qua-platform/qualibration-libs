"""
Unit tests for styling and theme functionality.
"""
import pytest
from unittest.mock import Mock, patch

from qualibration_libs.plotting import set_theme, set_palette, theme_context
from qualibration_libs.plotting.config import PlotTheme, RcParams, apply_theme_to_layout


class TestPlotTheme:
    """Test cases for PlotTheme class."""
    
    def test_default_theme(self):
        """Test default theme values."""
        theme = PlotTheme()
        
        assert theme.font_size == 14
        assert theme.title_size == 16
        assert theme.tick_label_size == 12
        assert theme.marker_size == 6
        assert theme.line_width == 2
        assert theme.show_grid is True
        assert theme.grid_opacity == 0.25
        assert theme.residuals_height_ratio == 0.35
        assert theme.figure_bg == "white"
        assert theme.paper_bg == "white"
        assert len(theme.colorway) == 10
    
    def test_custom_theme(self):
        """Test custom theme creation."""
        theme = PlotTheme(
            font_size=18,
            marker_size=8,
            show_grid=False,
            figure_bg="lightgray"
        )
        
        assert theme.font_size == 18
        assert theme.marker_size == 8
        assert theme.show_grid is False
        assert theme.figure_bg == "lightgray"
        # Other values should remain default
        assert theme.title_size == 16
        assert theme.line_width == 2


class TestRcParams:
    """Test cases for RcParams class."""
    
    def test_default_rc_params(self):
        """Test default RC parameters."""
        rc = RcParams()
        
        assert isinstance(rc.values, dict)
        assert len(rc.values) == 0
    
    def test_custom_rc_params(self):
        """Test custom RC parameters."""
        rc = RcParams()
        rc.values['showlegend'] = True
        rc.values['grid_opacity'] = 0.5
        
        assert rc.values['showlegend'] is True
        assert rc.values['grid_opacity'] == 0.5


class TestThemeFunctions:
    """Test cases for theme-related functions."""
    
    def test_set_theme_basic(self):
        """Test basic theme setting."""
        custom_theme = PlotTheme(font_size=20, marker_size=10)
        
        set_theme(theme=custom_theme)
        
        # Check that global theme was updated
        from qualibration_libs.plotting import config
        assert config.CURRENT_THEME.font_size == 20
        assert config.CURRENT_THEME.marker_size == 10
    
    def test_set_theme_palette_string(self):
        """Test setting palette with string."""
        set_palette("deep")
        
        from qualibration_libs.plotting import config
        assert config.CURRENT_PALETTE is not None
        assert len(config.CURRENT_PALETTE) > 0
    
    def test_set_theme_palette_custom(self):
        """Test setting custom palette."""
        custom_palette = ["#FF0000", "#00FF00", "#0000FF"]
        set_palette(custom_palette)
        
        from qualibration_libs.plotting import config
        assert config.CURRENT_PALETTE == tuple(custom_palette)
    
    def test_set_theme_rc_params(self):
        """Test setting RC parameters."""
        rc_params = {'showlegend': True, 'grid_opacity': 0.3}
        set_theme(rc=rc_params)
        
        from qualibration_libs.plotting import config
        assert config.CURRENT_RC.values['showlegend'] is True
        assert config.CURRENT_RC.values['grid_opacity'] == 0.3
    
    def test_set_theme_combined(self):
        """Test setting theme, palette, and RC together."""
        theme = PlotTheme(font_size=16)
        palette = ["#FF0000", "#00FF00"]
        rc = {'showlegend': False}
        
        set_theme(theme=theme, palette=palette, rc=rc)
        
        from qualibration_libs.plotting import config
        assert config.CURRENT_THEME.font_size == 16
        assert config.CURRENT_PALETTE == tuple(palette)
        assert config.CURRENT_RC.values['showlegend'] is False
    
    def test_theme_context(self):
        """Test theme context manager."""
        original_theme = PlotTheme()
        
        with theme_context(
            theme=PlotTheme(font_size=20),
            palette=["#FF0000"],
            rc={'test_param': True}
        ):
            from qualibration_libs.plotting import config
            assert config.CURRENT_THEME.font_size == 20
            assert config.CURRENT_PALETTE == ("#FF0000",)
            assert config.CURRENT_RC.values['test_param'] is True
        
        # Should restore original values
        from qualibration_libs.plotting import config
        assert config.CURRENT_THEME.font_size == original_theme.font_size
        assert config.CURRENT_PALETTE is None or config.CURRENT_PALETTE != ("#FF0000",)
        assert 'test_param' not in config.CURRENT_RC.values
    
    def test_theme_context_exception(self):
        """Test theme context manager with exception."""
        original_font_size = 14
        
        try:
            with theme_context(theme=PlotTheme(font_size=20)):
                from qualibration_libs.plotting import config
                assert config.CURRENT_THEME.font_size == 20
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Should still restore original values
        from qualibration_libs.plotting import config
        assert config.CURRENT_THEME.font_size == original_font_size


class TestApplyThemeToLayout:
    """Test cases for apply_theme_to_layout function."""
    
    def test_apply_theme_to_layout_dict(self):
        """Test applying theme to layout dictionary."""
        layout = {}
        theme = PlotTheme(font_size=18, figure_bg="lightblue")
        
        apply_theme_to_layout(layout)
        
        assert layout['template'] == "plotly_white"
        assert layout['paper_bgcolor'] == theme.paper_bg
        assert layout['plot_bgcolor'] == theme.figure_bg
        assert layout['font']['size'] == theme.font_size
    
    def test_apply_theme_to_layout_with_palette(self):
        """Test applying theme with palette."""
        from qualibration_libs.plotting import config
        
        # Set a palette
        config.CURRENT_PALETTE = ("#FF0000", "#00FF00", "#0000FF")
        
        layout = {}
        apply_theme_to_layout(layout)
        
        assert 'colorway' in layout
        assert layout['colorway'] == list(config.CURRENT_PALETTE)
    
    def test_apply_theme_to_layout_object(self):
        """Test applying theme to layout object with update method."""
        mock_layout = Mock()
        mock_layout.update = Mock()
        
        apply_theme_to_layout(mock_layout)
        
        # Should call update method
        mock_layout.update.assert_called_once()
        call_args = mock_layout.update.call_args[0][0]
        
        assert call_args['template'] == "plotly_white"
        assert 'paper_bgcolor' in call_args
        assert 'plot_bgcolor' in call_args
        assert 'font' in call_args


class TestBuiltInPalettes:
    """Test cases for built-in color palettes."""
    
    def test_qualibrate_palette(self):
        """Test qualibrate palette."""
        set_palette("qualibrate")
        
        from qualibration_libs.plotting import config
        assert config.CURRENT_PALETTE is not None
        assert len(config.CURRENT_PALETTE) == 10
    
    def test_deep_palette(self):
        """Test deep palette."""
        set_palette("deep")
        
        from qualibration_libs.plotting import config
        assert config.CURRENT_PALETTE is not None
        assert len(config.CURRENT_PALETTE) == 10
    
    def test_muted_palette(self):
        """Test muted palette."""
        set_palette("muted")
        
        from qualibration_libs.plotting import config
        assert config.CURRENT_PALETTE is not None
        assert len(config.CURRENT_PALETTE) == 10
    
    def test_unknown_palette(self):
        """Test unknown palette falls back to default."""
        set_palette("unknown_palette")
        
        from qualibration_libs.plotting import config
        # Should fall back to default theme colorway
        assert config.CURRENT_PALETTE is not None


class TestStyleIntegration:
    """Integration tests for styling system."""
    
    def test_theme_affects_plotting(self):
        """Test that theme settings affect plotting."""
        import numpy as np
        import xarray as xr
        from qualibration_libs.plotting import QualibrationFigure
        
        # Create test data
        data = xr.Dataset({
            'test': (['x'], np.random.rand(10))
        }, coords={'x': np.linspace(0, 10, 10)})
        
        # Set custom theme
        set_theme(theme=PlotTheme(font_size=20, marker_size=10))
        
        # Create plot
        fig = QualibrationFigure.plot(data, x='x', data_var='test')
        
        # Check that theme was applied
        assert fig.figure is not None
        # The actual theme application is tested in the figure tests
    
    def test_palette_affects_plotting(self):
        """Test that palette settings affect plotting."""
        import numpy as np
        import xarray as xr
        from qualibration_libs.plotting import QualibrationFigure
        
        # Create test data with multiple series
        data = xr.Dataset({
            'series1': (['x'], np.random.rand(10)),
            'series2': (['x'], np.random.rand(10))
        }, coords={'x': np.linspace(0, 10, 10)})
        
        # Set custom palette
        set_palette(["#FF0000", "#00FF00"])
        
        # Create plot
        fig = QualibrationFigure.plot(data, x='x', data_var='series1')
        
        # Check that palette was applied
        assert fig.figure is not None
        # The actual palette application is tested in the figure tests
    
    def test_rc_params_affect_plotting(self):
        """Test that RC parameters affect plotting."""
        import numpy as np
        import xarray as xr
        from qualibration_libs.plotting import QualibrationFigure
        
        # Create test data
        data = xr.Dataset({
            'test': (['x'], np.random.rand(10))
        }, coords={'x': np.linspace(0, 10, 10)})
        
        # Set RC parameters
        set_theme(rc={'showlegend': False})
        
        # Create plot
        fig = QualibrationFigure.plot(data, x='x', data_var='test')
        
        # Check that RC parameters were applied
        assert fig.figure is not None
        # The actual RC parameter application is tested in the figure tests


if __name__ == "__main__":
    pytest.main([__file__])
