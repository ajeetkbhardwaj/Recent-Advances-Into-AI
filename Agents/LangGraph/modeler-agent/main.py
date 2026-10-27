import json, pathlib
from src.graph import build_graph
from langgraph.types import Command

def main():
    graph = build_graph()
    print(graph.get_graph().draw_mermaid())
    q = input("\nBusiness question (e.g., 'Allocate 10k units across A/B/C to maximize profit with 95% service')\n> ").strip()
    if not q:
        q = "Allocate 10000 units across warehouses A/B/C to maximize profit (5,4,3 per unit) respecting capacities, ensure 95% service under stochastic demand."
    state = {"question": q, "iteration": 0, "messages": []}
    config = {"configurable": {"thread_id": "demo-001"}}
    result = graph.invoke(state, config=config)
    # handle interrupt
    snap = graph.get_state(config)
    if snap.next and "human_gate" in snap.next:
        print("\n[HITL] Low service level — approval needed")
        print(json.dumps(result.get("solver_result",{}), indent=2))
        print(json.dumps(result.get("simulation_result",{}), indent=2))
        try:
            ans = input("Approve? [y/n] (auto y in 5s): ").strip().lower()
        except EOFError:
            ans = "y"
        if ans == "":
            ans = "y"
        result = graph.invoke(Command(resume={"approved": ans=="y"}), config=config)
    print("\n" + "="*80 + "\nFINAL REPORT\n" + "="*80)
    print(result.get("final_report","(no report)"))
    pathlib.Path("reports/latest_report.md").write_text(result.get("final_report",""))
    print("\nSaved to reports/latest_report.md")
    print("Solver:", json.dumps(result.get("solver_result",{}), indent=2))
    print("Sim:", json.dumps(result.get("simulation_result",{}), indent=2))

if __name__ == "__main__":
    main()
