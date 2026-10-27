from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

class ModelerState(TypedDict, total=False):
    # user
    messages: Annotated[List[AnyMessage], add_messages]
    question: str

    # formulation
    problem_type: str  # lp | mip | stochastic | robust
    formulation: Dict[str, Any]  # {objective, constraints, variables}
    pulp_code: str
    valid: bool
    validation_errors: List[str]

    # solving
    solver_result: Dict[str, Any]  # {status, objective, allocation, raw}
    feasible: bool

    # simulation / risk
    simulation_result: Dict[str, Any]  # monte carlo, service_level
    sensitivity: Dict[str, Any]

    # synthesis
    findings: List[Dict[str, Any]]
    recommendations: List[str]
    final_report: str

    # control
    iteration: int
    steps: int
    errors: List[str]
    _needs_reformulation: bool
    _hint: str
