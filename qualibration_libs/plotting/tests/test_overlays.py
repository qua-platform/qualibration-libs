"""
Unit tests for overlay functionality.
"""
import pytest
import numpy as np
import plotly.graph_objects as go
from unittest.mock import Mock, patch

from qualibration_libs.plotting.overlays import (
    RefLine, LineOverlay, ScatterOverlay, TextBoxOverlay, FitOverlay
)


class TestRefLine:
    """Test cases for RefLine overlay."""
    
    def test_vertical_line(self):
        """Test vertical reference line."""
        ref_line = RefLine(x=5.0, name="Test Line")
        
        assert ref_line.x == 5.0
        assert ref_line.y is None
        assert ref_line.name == "Test Line"
        assert ref_line.dash == "dot"
    
    def test_horizontal_line(self):
        """Test horizontal reference line."""
        ref_line = RefLine(y=3.0, name="Horizontal")
        
        assert ref_line.x is None
        assert ref_line.y == 3.0
        assert ref_line.name == "Horizontal"
    
    def test_both_lines(self):
        """Test both vertical and horizontal lines."""
        ref_line = RefLine(x=5.0, y=3.0, name="Cross")
        
        assert ref_line.x == 5.0
        assert ref_line.y == 3.0
    
    def test_add_to_figure(self):
        """Test adding reference line to figure."""
        ref_line = RefLine(x=5.0, name="Test")
        
        # Mock figure and theme
        mock_fig = Mock()
        mock_theme = Mock()
        mock_theme.line_width = 2
        
        ref_line.add_to(mock_fig, row=1, col=1, theme=mock_theme)
        
        # Should call add_shape for vertical line
        mock_fig.add_shape.assert_called_once()
        call_args = mock_fig.add_shape.call_args
        assert call_args[1]['type'] == 'line'
        assert call_args[1]['x0'] == 5.0
        assert call_args[1]['x1'] == 5.0
        assert call_args[1]['y0'] == 0
        assert call_args[1]['y1'] == 1
        assert call_args[1]['xref'] == 'x'
        assert call_args[1]['yref'] == 'paper'
        assert call_args[1]['row'] == 1
        assert call_args[1]['col'] == 1


class TestLineOverlay:
    """Test cases for LineOverlay."""
    
    def test_basic_line(self):
        """Test basic line overlay creation."""
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        
        line = LineOverlay(x=x, y=y, name="Sine Wave")
        
        assert np.array_equal(line.x, x)
        assert np.array_equal(line.y, y)
        assert line.name == "Sine Wave"
        assert line.dash == "dash"
        assert line.show_legend is True
    
    def test_line_properties(self):
        """Test line overlay properties."""
        x = np.array([0, 1, 2])
        y = np.array([0, 1, 0])
        
        line = LineOverlay(
            x=x, y=y, name="Triangle",
            dash="solid", width=3.0, show_legend=False
        )
        
        assert line.dash == "solid"
        assert line.width == 3.0
        assert line.show_legend is False
    
    def test_add_to_figure(self):
        """Test adding line overlay to figure."""
        x = np.array([0, 1, 2])
        y = np.array([0, 1, 0])
        line = LineOverlay(x=x, y=y, name="Test Line")
        
        mock_fig = Mock()
        mock_theme = Mock()
        mock_theme.line_width = 2
        
        line.add_to(mock_fig, row=1, col=1, theme=mock_theme)
        
        # Should call add_trace
        mock_fig.add_trace.assert_called_once()
        call_args = mock_fig.add_trace.call_args
        trace = call_args[0][0]
        
        assert isinstance(trace, go.Scatter)
        assert trace.mode == "lines"
        assert trace.name == "Test Line"
        assert call_args[1]['row'] == 1
        assert call_args[1]['col'] == 1


class TestScatterOverlay:
    """Test cases for ScatterOverlay."""
    
    def test_basic_scatter(self):
        """Test basic scatter overlay creation."""
        x = np.array([1, 2, 3, 4])
        y = np.array([1, 4, 2, 3])
        
        scatter = ScatterOverlay(x=x, y=y, name="Points")
        
        assert np.array_equal(scatter.x, x)
        assert np.array_equal(scatter.y, y)
        assert scatter.name == "Points"
    
    def test_scatter_properties(self):
        """Test scatter overlay properties."""
        x = np.array([1, 2, 3])
        y = np.array([1, 2, 3])
        
        scatter = ScatterOverlay(
            x=x, y=y, name="Test Points",
            marker_size=10.0
        )
        
        assert scatter.marker_size == 10.0
    
    def test_add_to_figure(self):
        """Test adding scatter overlay to figure."""
        x = np.array([1, 2, 3])
        y = np.array([1, 2, 3])
        scatter = ScatterOverlay(x=x, y=y, name="Test Scatter")
        
        mock_fig = Mock()
        mock_theme = Mock()
        mock_theme.marker_size = 6
        
        scatter.add_to(mock_fig, row=1, col=1, theme=mock_theme)
        
        mock_fig.add_trace.assert_called_once()
        call_args = mock_fig.add_trace.call_args
        trace = call_args[0][0]
        
        assert isinstance(trace, go.Scatter)
        assert trace.mode == "markers"
        assert trace.name == "Test Scatter"
        assert call_args[1]['row'] == 1
        assert call_args[1]['col'] == 1


class TestTextBoxOverlay:
    """Test cases for TextBoxOverlay."""
    
    def test_basic_text_box(self):
        """Test basic text box creation."""
        text_box = TextBoxOverlay(text="Test Annotation")
        
        assert text_box.text == "Test Annotation"
        assert text_box.anchor == "top right"
    
    def test_anchor_positions(self):
        """Test different anchor positions."""
        # Top right
        tr_box = TextBoxOverlay(text="TR", anchor="top right")
        assert tr_box.anchor == "top right"
        
        # Bottom left
        bl_box = TextBoxOverlay(text="BL", anchor="bottom left")
        assert bl_box.anchor == "bottom left"
        
        # Top left
        tl_box = TextBoxOverlay(text="TL", anchor="top left")
        assert tl_box.anchor == "top left"
        
        # Bottom right
        br_box = TextBoxOverlay(text="BR", anchor="bottom right")
        assert br_box.anchor == "bottom right"
    
    def test_add_to_figure(self):
        """Test adding text box to figure."""
        text_box = TextBoxOverlay(text="Test Text", anchor="top right")
        
        mock_fig = Mock()
        mock_theme = Mock()
        
        text_box.add_to(mock_fig, row=1, col=1, theme=mock_theme)
        
        mock_fig.add_annotation.assert_called_once()
        call_args = mock_fig.add_annotation.call_args
        assert call_args[1]['text'] == "Test Text"
        assert call_args[1]['x'] == 1.0  # right
        assert call_args[1]['y'] == 1.0  # top
        assert call_args[1]['xref'] == 'paper'
        assert call_args[1]['yref'] == 'paper'
        assert call_args[1]['row'] == 1
        assert call_args[1]['col'] == 1


class TestFitOverlay:
    """Test cases for FitOverlay."""
    
    def test_basic_fit_overlay(self):
        """Test basic fit overlay creation."""
        x = np.linspace(0, 10, 100)
        y_fit = np.sin(x)
        params = {'amplitude': 1.0, 'frequency': 0.1}
        
        fit = FitOverlay(
            y_fit=y_fit,
            params=params,
            name="Fit Curve"
        )
        
        assert np.array_equal(fit.y_fit, y_fit)
        assert fit.params == params
        assert fit.name == "Fit Curve"
        assert fit.dash == "dash"
    
    def test_fit_with_formatter(self):
        """Test fit overlay with parameter formatter."""
        x = np.linspace(0, 10, 100)
        y_fit = np.sin(x)
        params = {'amplitude': 1.0, 'frequency': 0.1}
        
        def formatter(p):
            return f"A={p['amplitude']:.2f}, f={p['frequency']:.2f}"
        
        fit = FitOverlay(
            y_fit=y_fit,
            params=params,
            formatter=formatter,
            name="Formatted Fit"
        )
        
        assert fit.formatter is not None
        formatted_text = fit.formatter(params)
        assert "A=1.00" in formatted_text
        assert "f=0.10" in formatted_text
    
    def test_add_to_figure_with_curve(self):
        """Test adding fit overlay with curve to figure."""
        x = np.linspace(0, 10, 100)
        y_fit = np.sin(x)
        params = {'amplitude': 1.0}
        
        fit = FitOverlay(y_fit=y_fit, params=params, name="Sine Fit")
        
        mock_fig = Mock()
        mock_theme = Mock()
        mock_theme.line_width = 2
        
        fit.add_to(mock_fig, row=1, col=1, theme=mock_theme, x=x)
        
        # Should add trace for the fit curve
        mock_fig.add_trace.assert_called_once()
        call_args = mock_fig.add_trace.call_args
        trace = call_args[0][0]
        
        assert isinstance(trace, go.Scatter)
        assert trace.mode == "lines"
        assert trace.name == "Sine Fit"
        assert call_args[1]['row'] == 1
        assert call_args[1]['col'] == 1
    
    def test_add_to_figure_with_text(self):
        """Test adding fit overlay with text to figure."""
        params = {'amplitude': 1.0, 'frequency': 0.1}
        
        def formatter(p):
            return f"A={p['amplitude']:.2f}"
        
        fit = FitOverlay(params=params, formatter=formatter, name="Fit")
        
        mock_fig = Mock()
        mock_theme = Mock()
        
        # Mock the TextBoxOverlay.add_to method
        with patch('qualibration_libs.plotting.overlays.TextBoxOverlay.add_to') as mock_text_add:
            fit.add_to(mock_fig, row=1, col=1, theme=mock_theme)
            
            # Should call TextBoxOverlay.add_to
            mock_text_add.assert_called_once()
    
    def test_add_to_figure_no_curve_no_text(self):
        """Test adding fit overlay with neither curve nor text."""
        fit = FitOverlay(name="Empty Fit")
        
        mock_fig = Mock()
        mock_theme = Mock()
        
        fit.add_to(mock_fig, row=1, col=1, theme=mock_theme)
        
        # Should not add anything
        mock_fig.add_trace.assert_not_called()
        mock_fig.add_annotation.assert_not_called()


class TestOverlayIntegration:
    """Integration tests for overlay system."""
    
    def test_multiple_overlays(self):
        """Test combining multiple overlay types."""
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        
        overlays = [
            RefLine(x=5.0, name="Center"),
            LineOverlay(x=x, y=y, name="Sine"),
            ScatterOverlay(x=[2, 4, 6, 8], y=[0, 1, 0, -1], name="Points"),
            TextBoxOverlay(text="Test", anchor="top left")
        ]
        
        # Test that all overlays can be created
        assert len(overlays) == 4
        assert all(hasattr(overlay, 'add_to') for overlay in overlays)
    
    def test_overlay_style_override(self):
        """Test overlay style override functionality."""
        x = np.array([0, 1, 2])
        y = np.array([0, 1, 0])
        
        line = LineOverlay(x=x, y=y, name="Test")
        
        mock_fig = Mock()
        mock_theme = Mock()
        mock_theme.line_width = 2
        
        # Test with style override
        style_overrides = {
            'line': {'color': 'red', 'width': 5}
        }
        
        line.add_to(mock_fig, row=1, col=1, theme=mock_theme, **style_overrides)
        
        # Check that style overrides are applied
        call_args = mock_fig.add_trace.call_args
        trace = call_args[0][0]
        assert 'color' in str(trace.line)
        assert 'width' in str(trace.line)


if __name__ == "__main__":
    pytest.main([__file__])
