import sys
from pathlib import Path

import duckdb


def main() -> None:
    # Windows terminal encoding compatibility
    if sys.stdout.encoding != "utf-8" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

    root = Path(__file__).resolve().parents[2]
    traffic_path = root / "data" / "silver" / "traffic_hourly" / "*.parquet"
    weather_path = root / "data" / "silver" / "weather_hourly" / "*.parquet"
    gold_path = root / "data" / "gold" / "*.parquet"
    school_path = root / "data" / "silver" / "school" / "*.parquet"

    con = duckdb.connect()

    # Define convenient views for easier querying
    if list(traffic_path.parent.glob("*.parquet")):
        con.execute(
            f"CREATE VIEW traffic AS SELECT * FROM read_parquet('{traffic_path.as_posix()}')"
        )
    if list(weather_path.parent.glob("*.parquet")):
        con.execute(
            f"CREATE VIEW weather AS SELECT * FROM read_parquet('{weather_path.as_posix()}')"
        )
    if gold_path.parent.exists() and list(gold_path.parent.glob("*.parquet")):
        con.execute(
            f"CREATE VIEW main AS SELECT * FROM read_parquet('{gold_path.as_posix()}')"
        )
    if school_path.parent.exists() and list(school_path.parent.glob("*.parquet")):
        con.execute(
            f"CREATE VIEW school AS SELECT * FROM read_parquet('{school_path.as_posix()}')"
        )

    print("=" * 60)
    print("Istanbul City Pulse - Interactive DuckDB SQL Shell")
    print("=" * 60)
    print("Available Tables:")
    print("  - 'traffic' (2023-2025 hourly traffic data)")
    print("  - 'weather' (2023-2025 hourly weather data)")
    if gold_path.parent.exists() and list(gold_path.parent.glob("*.parquet")):
        print("  - 'main'    (Combined traffic + weather analytical table)")
    if school_path.parent.exists() and list(school_path.parent.glob("*.parquet")):
        print("  - 'school'  (School calendar features)")
    print(
        "\nTip: Multi-line SQL is supported. End with ';' and press Enter to execute."
    )
    print("To exit: 'exit', 'quit', or 'q'\n")

    buffer: list[str] = []

    while True:
        try:
            prompt = "duckdb> " if not buffer else "   ...> "
            line = input(prompt)
            stripped = line.strip()

            if not buffer:
                if not stripped:
                    continue
                if stripped.lower() in ("exit", "quit", "q"):
                    print("Exited console.")
                    break
                if stripped.lower() == "clear":
                    print("\n" * 50)
                    continue

            buffer.append(line)

            full_query = "\n".join(buffer).strip()
            if full_query.endswith(";") or (
                len(buffer) == 1
                and not stripped.upper().startswith(
                    ("SELECT", "WITH", "CREATE", "INSERT", "COPY")
                )
            ):
                con.sql(full_query.rstrip(";")).show()
                buffer.clear()

        except (KeyboardInterrupt, EOFError):
            print("\nExited console.")
            break
        except (duckdb.Error, ValueError) as e:
            print(f"Error: {e}\n")
            buffer.clear()


if __name__ == "__main__":
    main()
