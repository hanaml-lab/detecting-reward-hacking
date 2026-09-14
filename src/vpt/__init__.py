"""Verifier Perturbation Testing: trace-independent detection of reward hacking."""
from .detector import VPTDetector, VPTResult, conformal_threshold, RobustScale
from .types import Task, TestCase, Property, MetamorphicRelation, Workspace, Rollout

__version__ = "0.1.0"
__all__ = ["VPTDetector", "VPTResult", "conformal_threshold", "RobustScale", "Task", "TestCase",
           "Property", "MetamorphicRelation", "Workspace", "Rollout", "__version__"]
