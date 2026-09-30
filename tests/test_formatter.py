from src.data.filter import filter_records
from src.data.formatter import format_chatml


def test_format_chatml_contains_conversation_sections():
    result = format_chatml("CREATE TABLE users (id INTEGER);", "List users", "SELECT * FROM users")
    assert "<|im_start|>system" in result
    assert "SELECT * FROM users" in result


def test_filter_removes_duplicates_and_non_queries():
    records = [{"sql": "SELECT 1;"}, {"sql": " select 1 "}, {"sql": "DELETE FROM users"}]
    assert filter_records(records) == [records[0]]
