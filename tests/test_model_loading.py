import sqlite3

from src.evaluation.metrics import execution_accuracy


def test_execution_accuracy_compares_sqlite_results():
    connection = sqlite3.connect(":memory:")
    assert execution_accuracy(["SELECT 1"], ["SELECT 1"], connection) == 1.0
    assert execution_accuracy(["SELECT 2"], ["SELECT 1"], connection) == 0.0
