"""
Custom exceptions for the plotting module.

This module defines a hierarchy of exceptions for better error handling and
debugging in the quantum calibration plotting framework.
"""

from typing import Dict, List, Optional, Any


class PlottingError(Exception):
    """
    Base exception for all plotting-related errors.
    
    This exception provides additional context and suggestions to help
    users debug issues more effectively.
    
    Attributes:
        message: The error message
        context: Additional context about the error (e.g., available values)
        suggestions: List of suggestions for fixing the error
    """
    
    def __init__(
        self, 
        message: str, 
        context: Optional[Dict[str, Any]] = None, 
        suggestions: Optional[List[str]] = None
    ):
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.suggestions = suggestions or []
    
    def __str__(self) -> str:
        """Format the error message with context and suggestions."""
        parts = [self.message]
        
        if self.context:
            parts.append("\nContext:")
            for key, value in self.context.items():
                parts.append(f"  - {key}: {value}")
        
        if self.suggestions:
            parts.append("\nSuggestions:")
            for i, suggestion in enumerate(self.suggestions, 1):
                parts.append(f"  {i}. {suggestion}")
        
        return "\n".join(parts)


class ConfigurationError(PlottingError):
    """Raised when plot configuration is invalid or incompatible."""
    pass


class DataSourceError(PlottingError):
    """Raised when required data sources are missing or invalid."""
    pass


class EngineError(PlottingError):
    """Raised when a rendering engine encounters an error."""
    pass


class ExperimentTypeError(PlottingError):
    """Raised when experiment type cannot be determined or is unsupported."""
    pass


class QubitError(PlottingError):
    """Raised when qubit-related operations fail."""
    pass


class OverlayError(PlottingError):
    """Raised when overlay rendering fails."""
    pass


class FitDataError(PlottingError):
    """Raised when fit data is missing, invalid, or incompatible."""
    pass


class ValidationError(PlottingError):
    """Raised when data validation fails."""
    pass


class DimensionError(PlottingError):
    """Raised when data dimensions are incompatible with the requested plot type."""
    pass


class ColorScaleError(PlottingError):
    """Raised when colorscale calculations fail."""
    pass