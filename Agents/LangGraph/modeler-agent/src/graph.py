from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
# for prod swap: from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command
from .state import ModelerState
from . import nodes

def route_after_validate(state: ModelerState):
    return "solver" if state.get("valid", True) else "formulate"

def route_after_solver(state: ModelerState):
    return "simulator" if state.get("feasible") else "challenger"

def route_after_challenger(state: ModelerState):
    if state.get("_needs_reformulation") and state.get("iteration",0) < 3:
        return "reformulate"
    # human gate for high objective / low service
    sim = state.get("simulation_result",{})
    if sim.get("service_level",1) < 0.95:
        return "human_gate"
    return "report"

def human_gate(state: ModelerState):
    # pause for approval — respects checkpointer
    decision = interrupt({"question":"Approve allocation?","solver": state.get("solver_result"), "simulation": state.get("simulation_result")})
    if not decision.get("approved", True):
        return {"recommendations": state.get("recommendations",[])+["Human rejected — reformulate with lower risk"]}
    return {}

def build_graph(checkpointer=None):
    if checkpointer is None:
        checkpointer = MemorySaver()
    g = StateGraph(ModelerState)
    g.add_node("formulate", nodes.formulate)
    g.add_node("validate", nodes.validate_node)
    g.add_node("solver", nodes.solver)
    g.add_node("simulator", nodes.simulator)
    g.add_node("sensitivity", nodes.sensitivity)
    g.add_node("challenger", nodes.challenger)
    g.add_node("reformulate", nodes.reformulate)
    g.add_node("human_gate", human_gate)
    g.add_node("report", nodes.report)

    g.add_edge(START, "formulate")
    g.add_edge("formulate", "validate")
    g.add_conditional_edges("validate", route_after_validate, {"solver":"solver","formulate":"formulate"})
    g.add_conditional_edges("solver", route_after_solver, {"simulator":"simulator","challenger":"challenger"})
    g.add_edge("simulator", "sensitivity")
    g.add_edge("sensitivity", "challenger")
    g.add_conditional_edges("challenger", route_after_challenger, {"reformulate":"reformulate","human_gate":"human_gate","report":"report"})
    g.add_edge("reformulate", "solver")
    g.add_edge("human_gate", "report")
    g.add_edge("report", END)
    return g.compile(checkpointer=checkpointer)
