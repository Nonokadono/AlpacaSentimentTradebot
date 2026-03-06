# CHANGES:
# - Added focused regression tests for the confirmed New-Strat failures:
#   1) opening baseline saved from sentiment instead of technical composite,
#   2) missing stop-loss / take-profit protected entry path,
#   3) destructive purge of opening baseline on failed close,
#   4) missing flat-position confirmation before state purge,
#   5) missing network guard around main loop broker bootstrap.
# - HARDENED the protected-entry regression test by stubbing execution.order_executor.log_proposed_trade
#   to a no-op, so the test validates execution behavior directly and cannot fail because of unrelated
#   logging-field drift.
# - The fixture still provides the key ProposedTrade-like fields used by the executor path.
# - This file is self-contained and runnable with: python test.py

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
MAIN_PATH = ROOT / "main.py"
ORDER_EXECUTOR_PATH = ROOT / "execution" / "order_executor.py"
ADAPTER_PATH = ROOT / "adapters" / "alpaca_adapter.py"


class ContractFailure(AssertionError):
    pass


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise ContractFailure(message)


def read_text(path: Path) -> str:
    assert_true(path.exists(), f"Missing required file: {path}")
    return path.read_text(encoding="utf-8", errors="ignore")


def parse_ast(path: Path) -> ast.AST:
    return ast.parse(read_text(path), filename=str(path))


def get_function_node(module: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise ContractFailure(f"Function not found: {name}")


def extract_source_segment(path: Path, func_name: str) -> str:
    text = read_text(path)
    module = ast.parse(text, filename=str(path))
    node = get_function_node(module, func_name)
    segment = ast.get_source_segment(text, node)
    if not segment:
        raise ContractFailure(f"Could not extract source for {func_name}")
    return segment


def test_repo_layout() -> None:
    assert_true(MAIN_PATH.exists(), "main.py not found.")
    assert_true(ORDER_EXECUTOR_PATH.exists(), "execution/order_executor.py not found.")
    assert_true(ADAPTER_PATH.exists(), "adapters/alpaca_adapter.py not found.")


def test_main_saves_opening_compound_from_signal_score() -> None:
    text = read_text(MAIN_PATH)

    good = "_opening_compounds[proposed.symbol] = proposed.signal_score"
    bad = "_opening_compounds[proposed.symbol] = proposed.sentiment_score"

    assert_true(good in text, "main.py must persist proposed.signal_score as the opening baseline.")
    assert_true(bad not in text, "main.py still persists proposed.sentiment_score as the opening baseline.")


def test_entry_uses_protective_orders_when_stop_and_tp_exist() -> None:
    import execution.order_executor as order_executor_module
    from execution.order_executor import OrderExecutor

    order_executor_module.log_proposed_trade = lambda *args, **kwargs: None

    class FakeAdapter:
        def __init__(self) -> None:
            self.calls: list[tuple] = []

        def list_orders(self, status: str = "open"):
            self.calls.append(("list_orders", status))
            return []

        def cancel_order(self, order_id):
            self.calls.append(("cancel_order", order_id))

        def submit_bracket_order(self, **kwargs):
            self.calls.append(("submit_bracket_order", kwargs))
            return SimpleNamespace(id="bracket-1", **kwargs)

        def submit_market_order(self, **kwargs):
            self.calls.append(("submit_market_order", kwargs))
            return SimpleNamespace(id="market-1", **kwargs)

        def submit_stop_order(self, **kwargs):
            self.calls.append(("submit_stop_order", kwargs))
            return SimpleNamespace(id="stop-1", **kwargs)

        def submit_take_profit_limit_order(self, **kwargs):
            self.calls.append(("submit_take_profit_limit_order", kwargs))
            return SimpleNamespace(id="tp-1", **kwargs)

        def submit_trailing_stop_order(self, **kwargs):
            self.calls.append(("submit_trailing_stop_order", kwargs))
            return SimpleNamespace(id="trail-1", **kwargs)

        def get_position(self, symbol: str):
            self.calls.append(("get_position", symbol))
            return SimpleNamespace(symbol=symbol)

    execution_cfg = SimpleNamespace(
        enable_take_profit=True,
        enable_trailing_stop=False,
        entry_time_in_force="day",
        exit_time_in_force="gtc",
        trailing_stop_percent=2.0,
        post_entry_fill_poll_timeout_sec=1.0,
        post_entry_fill_poll_interval_sec=0.01,
    )

    adapter = FakeAdapter()
    executor = OrderExecutor(
        adapter=adapter,
        env_mode="PAPER",
        live_trading_enabled=True,
        execution_cfg=execution_cfg,
    )

    proposed = SimpleNamespace(
        symbol="AMD",
        qty=10.0,
        side="buy",
        stop_price=95.0,
        take_profit_price=110.0,
        rejected_reason=None,
        sentiment_score=0.25,
        signal_score=0.78,
        entry_price=100.0,
        risk_amount=50.0,
        risk_pct_of_equity=0.005,
        sentiment_scale=1.0,
    )

    order = executor.execute_proposed_trade(proposed)
    call_names = [name for name, *_ in adapter.calls]

    assert_true(order is not None, "Protected entry should return an order object.")
    assert_true(
        "submit_bracket_order" in call_names or (
            "submit_stop_order" in call_names and "submit_take_profit_limit_order" in call_names
        ),
        "Entry path did not create stop-loss / take-profit protection when stop_price and take_profit_price were available."
    )
    assert_true(
        not ("submit_market_order" in call_names and "submit_bracket_order" not in call_names and "submit_stop_order" not in call_names),
        "Entry path fell back to a naked market order instead of protected execution."
    )


def test_close_failure_does_not_purge_opening_baseline() -> None:
    from execution.order_executor import OrderExecutor

    class FailingCloseAdapter:
        def __init__(self) -> None:
            self.calls: list[tuple] = []

        def list_orders(self, status: str = "open"):
            self.calls.append(("list_orders", status))
            return []

        def cancel_order(self, order_id):
            self.calls.append(("cancel_order", order_id))

        def submit_market_order(self, **kwargs):
            self.calls.append(("submit_market_order", kwargs))
            raise RuntimeError("broker close failed")

    persisted_snapshots: list[dict] = []

    def persist_opening_compounds(snapshot):
        persisted_snapshots.append(dict(snapshot))

    execution_cfg = SimpleNamespace(
        enable_take_profit=True,
        enable_trailing_stop=False,
        entry_time_in_force="day",
        exit_time_in_force="gtc",
        trailing_stop_percent=2.0,
        post_entry_fill_poll_timeout_sec=1.0,
        post_entry_fill_poll_interval_sec=0.01,
    )

    adapter = FailingCloseAdapter()
    executor = OrderExecutor(
        adapter=adapter,
        env_mode="LIVE",
        live_trading_enabled=True,
        execution_cfg=execution_cfg,
    )

    position = SimpleNamespace(symbol="AMD", side="long", qty=10.0)
    sentiment = SimpleNamespace(score=-0.7, confidence=0.9, explanation="bad news")
    opening_compounds = {"AMD": 0.42}

    executor.close_position_due_to_sentiment(
        position=position,
        sentiment=sentiment,
        reason="soft_exit",
        env_mode="LIVE",
        opening_compounds=opening_compounds,
        persist_opening_compounds=persist_opening_compounds,
    )

    assert_true(
        "AMD" in opening_compounds,
        "Opening baseline was purged even though the close order failed."
    )
    assert_true(
        len(persisted_snapshots) == 0,
        "Persistent state was updated even though the close order failed."
    )


def test_close_path_requires_flat_confirmation_before_purge() -> None:
    source = extract_source_segment(ORDER_EXECUTOR_PATH, "close_position_due_to_sentiment").lower()

    assert_true(
        ("get_position(" in source) or ("list_positions(" in source) or ("_wait_for_flat" in source),
        "close_position_due_to_sentiment() must confirm the position is flat before purging entry state."
    )

    purge_idx = source.find("del opening_compounds[position.symbol]")
    verify_candidates = [
        source.find("get_position("),
        source.find("list_positions("),
        source.find("_wait_for_flat"),
    ]
    verify_candidates = [idx for idx in verify_candidates if idx != -1]

    assert_true(
        bool(verify_candidates),
        "No flat-confirmation call found in close_position_due_to_sentiment()."
    )
    assert_true(
        min(verify_candidates) < purge_idx if purge_idx != -1 else False,
        "Entry baseline purge appears before flat-position confirmation."
    )


def test_main_loop_handles_broker_connection_errors() -> None:
    module = parse_ast(MAIN_PATH)
    main_fn = get_function_node(module, "main")
    main_text = read_text(MAIN_PATH)

    found_guarded_get_account = False

    class GuardVisitor(ast.NodeVisitor):
        def visit_Try(self, node: ast.Try):
            nonlocal found_guarded_get_account
            segment = ast.get_source_segment(main_text, node) or ""
            if "adapter.get_account(" in segment:
                found_guarded_get_account = True
            self.generic_visit(node)

    GuardVisitor().visit(main_fn)

    assert_true(
        found_guarded_get_account,
        "main() does not guard adapter.get_account() with try/except recovery."
    )

    main_text_lower = main_text.lower()
    assert_true(
        ("time.sleep(" in main_text_lower and "continue" in main_text_lower) or "retry" in main_text_lower,
        "main() should back off and continue after broker/network errors."
    )


def run() -> int:
    tests = [
        test_repo_layout,
        test_main_saves_opening_compound_from_signal_score,
        test_entry_uses_protective_orders_when_stop_and_tp_exist,
        test_close_failure_does_not_purge_opening_baseline,
        test_close_path_requires_flat_confirmation_before_purge,
        test_main_loop_handles_broker_connection_errors,
    ]

    failures: list[str] = []

    print("Running New-Strat focused regression tests...\n")
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except Exception as exc:
            failures.append(f"{test.__name__}: {exc}")
            print(f"FAIL  {test.__name__}\n      {exc}")

    print("\n" + ("-" * 72))
    if failures:
        print(f"FAILED: {len(failures)} test(s) did not pass.")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("SUCCESS: all focused regression tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
