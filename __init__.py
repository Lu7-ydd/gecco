"""
GeCCo: Gene Co-expression Constructed identity framework
========================================================

A gene-centric framework that constructs cell identities as emergent entities 
shaped by mutually supportive gene programs.
"""

from .core import GeCCo
from .hierarchy import GeneHierarchy, GeneNode

__version__ = "0.1.0"
__author__ = "Luqi Yang"

# 定义 from gecco import * 时会导入什么
__all__ = ["GeCCo", "GeneHierarchy", "GeneNode"]