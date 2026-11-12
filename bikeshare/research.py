"""Compare calendar-only and weather-aware retrospective demand estimates."""
import json
from typing import TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bikeshare.pipeline import OUTPUT


class State(TypedDict, total=False):
    frame: pd.DataFrame
    adaptive: bool
    fail_first: bool
    trial: str
    retried: bool
    error: str
    scores: dict
    result: dict
    route: str


def prepare(state):
    frame = state["frame"].copy()
    dates = pd.to_datetime(frame.date)
    if not dates.is_monotonic_increasing or dates.duplicated().any():
        raise ValueError("Expected unique, chronological dates")
    frame["weekday"] = dates.dt.dayofweek
    frame["month"] = dates.dt.month
    frame["trend"] = (dates - dates.iloc[0]).dt.days
    frame = pd.get_dummies(frame, columns=["weekday", "month"], dtype=int)
    return {"frame": frame, "scores": {}}


def score(frame, trial, holdout=False):
    weather = ["tmin_c", "tmax_c", "rain_mm"]
    columns = [c for c in frame if c not in ["date", "rentals", *weather]]
    if trial != "calendar":
        columns += weather
    model = (RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42)
             if trial == "forest" else make_pipeline(StandardScaler(), Ridge(alpha=5)))
    train_end, validation_end = int(len(frame) * .6), int(len(frame) * .8)
    end = validation_end if holdout else train_end
    fit = frame.iloc[:end]
    check = frame.iloc[end:] if holdout else frame.iloc[end:validation_end]
    model.fit(fit[columns], fit.rentals)
    return float(mean_absolute_error(check.rentals, model.predict(check[columns]).clip(min=0)))


def planner(state):
    scores = state["scores"]
    if not scores:
        return {"trial": "calendar"}
    if "weather" not in scores:
        return {"trial": "weather"}
    return {"trial": "forest"}


def specialist(state):
    trial = state["trial"]
    try:
        if trial == "weather" and state.get("fail_first") and not state.get("retried"):
            raise RuntimeError("Controlled transient training error")
        value = score(state["frame"], trial)
    except RuntimeError as exc:
        return {"error": str(exc)}
    return {"scores": {**state["scores"], trial: value}, "error": ""}


def reviewer(state):
    if state["error"]:
        return {"route": "retry" if state["adaptive"] and not state.get("retried") else "stop"}
    scores = state["scores"]
    if state["trial"] == "calendar":
        return {"route": "plan"}
    if (state["adaptive"] and state["trial"] == "weather"
            and scores["weather"] >= .9 * scores["calendar"]):
        return {"route": "plan"}
    return {"route": "holdout"}


def recovery(state):
    return {"retried": True}


def holdout(state):
    best = min(state["scores"], key=state["scores"].get)
    return {"result": {"selected": best, "validation_mae": state["scores"],
                       "holdout_mae": score(state["frame"], best, holdout=True),
                       "retried": state.get("retried", False)}}


def graph():
    workflow = StateGraph(State)
    for node in (prepare, planner, specialist, reviewer, recovery, holdout):
        workflow.add_node(node.__name__, node)
    workflow.add_edge(START, "prepare")
    workflow.add_edge("prepare", "planner")
    workflow.add_edge("planner", "specialist")
    workflow.add_edge("specialist", "reviewer")
    workflow.add_conditional_edges("reviewer", lambda state: state["route"],
                                   {"retry": "recovery", "plan": "planner",
                                    "stop": END, "holdout": "holdout"})
    workflow.add_edge("recovery", "specialist")
    workflow.add_edge("holdout", END)
    return workflow.compile()


def evaluate():
    frame = pd.read_csv(OUTPUT / "daily.csv")
    prepared = prepare({"frame": frame})["frame"]
    app = graph()
    simple = app.invoke({"frame": frame, "adaptive": False})["result"]
    adaptive = app.invoke({"frame": frame, "adaptive": True})["result"]
    recovered = app.invoke({"frame": frame, "adaptive": True, "fail_first": True})["result"]
    fixed = {"selected": "calendar", "holdout_mae": score(prepared, "calendar", holdout=True)}
    cases = [
        {"input": "Chronological joined daily records",
         "expected": "Finite holdout score and weather model evaluated",
         "criterion": "Holdout MAE is finite and nonnegative, weather trial recorded",
         "threshold": 1,
         "score": int(0 <= adaptive["holdout_mae"] < float("inf")
                      and "weather" in adaptive["validation_mae"])},
        {"input": "Controlled first weather training failure",
         "expected": "One recovery and same final MAE",
         "criterion": "Retry flag set and final MAE matches clean run",
         "threshold": 1,
         "score": int(recovered["retried"] and recovered["holdout_mae"] == adaptive["holdout_mae"])},
        {"input": "Calendar MAE 100, weather MAE 95",
         "expected": "Evaluate forest specialist next",
         "criterion": "Reviewer routes back to planner",
         "threshold": 1,
         "score": int(reviewer({"error": "", "adaptive": True, "trial": "weather",
                                "scores": {"calendar": 100, "weather": 95}})["route"] == "plan")},
        {"input": "Training error after recovery already attempted",
         "expected": "Stop instead of retrying again",
         "criterion": "Reviewer chooses stop",
         "threshold": 1,
         "score": int(reviewer({"error": "controlled error", "adaptive": True,
                                "retried": True})["route"] == "stop")},
    ]
    for case in cases:
        case["passed"] = case["score"] >= case["threshold"]
    result = {"rows": len(frame), "train": int(len(frame) * .6),
              "validation": int(len(frame) * .8) - int(len(frame) * .6),
              "holdout": len(frame) - int(len(frame) * .8),
              "fixed": fixed, "simple": simple, "adaptive": adaptive,
              "recovery": recovered, "cases": cases}
    (OUTPUT / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    if not all(case["passed"] for case in cases):
        raise AssertionError("Evaluation case failed")
    return result


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
