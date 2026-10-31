"""Terminal içi interaktif DuckDB SQL konsolu.

Kullanım:
    uv run python -m city_pulse.sql_shell
    BİZ make shell yapıyoruz.
"""

import sys
from pathlib import Path

import duckdb


def main() -> None:
    # Windows terminal encoding uyumluluğu
    if sys.stdout.encoding != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass

    root = Path(__file__).resolve().parents[2]
    traffic_path = root / "data" / "silver" / "traffic_hourly" / "*.parquet"
    weather_path = root / "data" / "silver" / "weather_hourly" / "*.parquet"
    gold_path = root / "data" / "gold" / "*.parquet"

    con = duckdb.connect()

    # Kullanıcı uzun parquet yolları yazmakla uğraşmasın diye hazır view'lar tanımlıyoruz:
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

    print("=" * 60)
    print("Istanbul City Pulse - Interaktif DuckDB SQL Konsolu")
    print("=" * 60)
    print("Hazir Tablolar:")
    print("  - 'traffic' (2023-2025 saatlik trafik verisi)")
    print("  - 'weather' (2023-2025 saatlik hava durumu verisi)")
    if gold_path.parent.exists() and list(gold_path.parent.glob("*.parquet")):
        print("  - 'main'    (Trafik + Hava Durumu birlesik analitik tablosu)")
    print("\nIpucu: Cok satirli SQL yazabilirsiniz. Calistirmak icin sonuna ';' koyup Enter'a basin.")
    print("Cikmak icin: 'exit', 'quit' veya 'q'\n")

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
                    print("Konsoldan cikildi.")
                    break
                if stripped.lower() == "clear":
                    print("\n" * 50)
                    continue

            buffer.append(line)

            # Eger satir ';' ile bitiyorsa veya tek basina calisabilecek basit komutsa
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
            print("\nKonsoldan cikildi.")
            break
        except (duckdb.Error, ValueError) as e:
            print(f"Hata: {e}\n")
            buffer.clear()


if __name__ == "__main__":
    main()
