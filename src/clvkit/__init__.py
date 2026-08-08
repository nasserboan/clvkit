"""clvkit — CLV engine and cohort-retention over a shared transaction-log contract."""

from importlib.metadata import version as _version

from clvkit.clv import BGNBD, CLV, MBGNBD, GammaGamma
from clvkit.cohort import CohortMatrix, CohortSurvival
from clvkit.customer_base import CustomerBase

__version__ = _version("clvkit")

__all__ = [
    "BGNBD",
    "CLV",
    "MBGNBD",
    "CohortMatrix",
    "CohortSurvival",
    "CustomerBase",
    "GammaGamma",
]
